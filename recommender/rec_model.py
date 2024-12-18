import warnings
from typing import Dict, Optional
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from baselines import constants as cts

from torch import nn
from torch.nn import CrossEntropyLoss
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel

# Suppress FutureWarnings from transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]

        :return: A 3d tensor with shape of (N, T_q, C)

        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)

        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        matmul_output = (
            torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size**0.5
        )  # (h*N, T_q, T_k)

        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(
            self.num_heads, 1
        )  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(
            1, queries.shape[1], 1
        )  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-(2**32) + 1)
        matmul_output_m1 = torch.where(
            torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output
        )  # (h*N, T_q, T_k)

        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])  # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(
            matmul_output.shape[0], 1, 1
        )  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-(2**32) + 1)
        matmul_output_m2 = torch.where(
            torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1
        )  # (h*N, T_q, T_k)

        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)

        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(
            self.num_heads, 1
        )  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(
            1, 1, keys.shape[1]
        )  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask

        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)

        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)

        # Restore Shape
        output = torch.cat(
            torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2
        )  # (N, T_q, C)

        # Residual Connection
        output_res = output + queries

        return output_res


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


class GRU(nn.Module):
    def __init__(
        self,
        hidden_size,
        item_num,
        state_size,
        gru_layers=1,
        extra_embeddings=None,
        no_id=False,
    ):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.no_id = no_id
        if no_id:
            self.input_hidden_size = 0
        else:
            self.input_hidden_size = hidden_size
        if extra_embeddings is not None:
            # load extra embeddings if provided
            extra_emb = torch.load(extra_embeddings)
            self.input_hidden_size += extra_emb.shape[-1]
            if extra_emb.shape[0] == self.item_num:
                extra_emb = torch.cat(
                    [
                        extra_emb,
                        torch.zeros((1, extra_emb.shape[-1])).to(extra_emb.device),
                    ],
                    dim=0,
                )
            assert extra_emb.shape[0] == self.item_num + 1
            self.input_fc = nn.Linear(self.input_hidden_size, hidden_size)
            self.extra_emb = extra_emb
        else:
            self.extra_emb = None

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num + 1)

    def forward(self, batch):
        states = batch[cts.SEQUENCE]
        len_states = batch[cts.LEN_SEQ]
        # Supervised Head
        len_states = len_states.cpu()
        inputs_emb = self.item_embeddings(states)
        if self.extra_emb is not None:
            extra_emb = self.extra_emb[states]
            if self.no_id:
                inputs_emb = extra_emb
            else:
                inputs_emb = torch.cat([inputs_emb, extra_emb], dim=-1)
            inputs_emb = self.input_fc(inputs_emb)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            inputs_emb, len_states, batch_first=True, enforce_sorted=False
        )
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output


class Caser(nn.Module):
    def __init__(
        self,
        hidden_size,
        item_num,
        state_size,
        num_filters,
        filter_sizes,
        dropout,
        extra_embeddings=None,
        no_id=False,
    ):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout
        self.no_id = no_id
        if no_id:
            self.input_hidden_size = 0
        else:
            self.input_hidden_size = hidden_size
        if extra_embeddings is not None:
            # load extra embeddings if provided
            extra_emb = torch.load(extra_embeddings)
            self.input_hidden_size += extra_emb.shape[-1]
            if extra_emb.shape[0] == self.item_num:
                extra_emb = torch.cat(
                    [
                        extra_emb,
                        torch.zeros((1, extra_emb.shape[-1])).to(extra_emb.device),
                    ],
                    dim=0,
                )
            assert extra_emb.shape[0] == self.item_num + 1
            self.input_fc = nn.Linear(self.input_hidden_size, hidden_size)
            self.extra_emb = extra_emb
        else:
            self.extra_emb = None

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [
                nn.Conv2d(1, self.num_filters, (i, self.hidden_size))
                for i in self.filter_sizes
            ]
        )
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num + 1)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, batch):
        states = batch[cts.SEQUENCE]
        inputs_emb = self.item_embeddings(states)
        if self.extra_emb is not None:
            extra_emb = self.extra_emb[states]
            if self.no_id:
                inputs_emb = extra_emb
            else:
                inputs_emb = torch.cat([inputs_emb, extra_emb], dim=-1)
            inputs_emb = self.input_fc(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1)
        inputs_emb *= mask
        inputs_emb = inputs_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(inputs_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(inputs_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output


class SASRec(nn.Module):
    def __init__(
        self,
        hidden_size,
        item_num,
        state_size,
        dropout,
        device,
        num_heads=1,
        extra_embeddings=None,
        no_id=False,
    ):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.no_id = no_id
        if no_id:
            self.input_hidden_size = 0
        else:
            self.input_hidden_size = hidden_size
        if extra_embeddings is not None:
            # load extra embeddings if provided
            extra_emb = torch.load(extra_embeddings)
            self.input_hidden_size += extra_emb.shape[-1]
            if extra_embeddings.shape[0] == self.item_num:
                extra_embeddings = torch.cat(
                    [
                        extra_emb,
                        torch.zeros((1, extra_emb.shape[-1])).to(extra_emb.device),
                    ],
                    dim=0,
                )
            assert extra_emb.shape[0] == self.item_num + 1
            self.input_fc = nn.Linear(self.input_hidden_size, hidden_size)
            self.extra_emb = extra_emb
        else:
            self.extra_emb = None

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size, embedding_dim=hidden_size
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num + 1)

    def forward(self, batch):
        states = batch[cts.SEQUENCE]
        len_states = batch[cts.LEN_SEQ]

        inputs_emb = self.item_embeddings(states)
        if self.extra_emb is not None:
            extra_emb = self.extra_emb[states]
            if self.no_id:
                inputs_emb = extra_emb
            else:
                inputs_emb = torch.cat([inputs_emb, extra_emb], dim=-1)
            inputs_emb = self.input_fc(inputs_emb)
        inputs_emb += self.positional_embeddings(
            torch.arange(self.state_size).to(self.device)
        )
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output


class RecformerForSeqRec(LongformerPreTrainedModel):
    def __init__(
        self,
        model_name: str,
        max_attr_num: int,
        max_attr_length: int,
        max_item_embeddings: int,
        attention_window: str,
        max_token_num: int,
        item_num: int,
        pretrain_ckpt: str,
        extra_embeddings: Optional[str] = None,
    ):
        try:
            from recommender.RecFormer.recformer import (
            RecformerConfig,
            RecformerModel,
            Similarity,
        )      
        except ImportError:
            raise ImportError("Recformer submodule not inialized. Please initialize the submodule by running `git submodule update --init --recursive`")
            
        # Initialize the configuration
        config = RecformerConfig.from_pretrained(model_name)
        config.max_attr_num = max_attr_num
        config.max_attr_length = max_attr_length
        config.max_item_embeddings = max_item_embeddings
        config.attention_window = eval(attention_window)
        config.max_token_num = max_token_num
        config.item_num = item_num

        # Initialize the superclass with the configuration
        super().__init__(config)

        # Load the pretrained checkpoint
        pretrain_ckpt = torch.load(pretrain_ckpt)
        self.load_state_dict(pretrain_ckpt, strict=False)

        # Initialize model components
        self.longformer = RecformerModel(config)
        self.sim = Similarity(config)

        # Initialize item embeddings if needed
        if extra_embeddings:
            item_embeddings = torch.load(extra_embeddings)
            self.init_item_embedding(item_embeddings)
        else:
            self.init_item_embedding(None)
        self.post_init()

    def init_item_embedding(self, embeddings: str = None):
        self.item_embedding = nn.Embedding(
            num_embeddings=self.config.item_num, embedding_dim=self.config.hidden_size
        )
        if embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
            print("Initalize item embeddings from vectors.")

    def similarity_score(self, pooler_output, candidates=None):
        if candidates is None:
            candidate_embeddings = self.item_embedding.weight.unsqueeze(
                0
            )  # (1, num_items, hidden_size)
        else:
            candidate_embeddings = self.item_embedding(
                candidates
            )  # (batch_size, candidates, hidden_size)
        pooler_output = pooler_output.unsqueeze(1)  # (batch_size, 1, hidden_size)

        return self.sim(pooler_output, candidate_embeddings)

    def forward(self, batch: Dict[str, Optional[torch.Tensor]]):
        input_ids = batch.get("input_ids", None)
        attention_mask = batch.get("attention_mask", None)
        global_attention_mask = batch.get("global_attention_mask", None)
        head_mask = batch.get("head_mask", None)
        token_type_ids = batch.get("token_type_ids", None)
        position_ids = batch.get("position_ids", None)
        item_position_ids = batch.get("item_position_ids", None)
        inputs_embeds = batch.get("inputs_embeds", None)
        output_attentions = batch.get("output_attentions", None)
        output_hidden_states = batch.get("output_hidden_states", None)
        return_dict = batch.get("return_dict", None)

        candidates = batch.get(cts.CANDIDATES, None)  # candidate item ids
        labels = batch.get(cts.LABEL, None)  # target item ids

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooler_output = outputs.pooler_output  # (bs, hidden_size)

        if labels is None:
            return self.similarity_score(pooler_output, candidates)

        loss_fct = CrossEntropyLoss()

        if self.config.finetune_negative_sample_size <= 0:  ## using full softmax
            logits = self.similarity_score(pooler_output)
            loss = loss_fct(logits, labels)

        else:  ## using sampled softmax
            logits = self.similarity_score(pooler_output, candidates)
            target = torch.zeros_like(labels, device=labels.device)
            loss = loss_fct(logits, target)

        return loss


class LammaRec:
    def __init__(self, load_in_4bit: bool, bnb_4bit_use_double_quant: bool):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        model.config.use_cache = False

        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids, attention_mask).logits
