from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn

from transformers import PreTrainedModel, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging

from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

logger = logging.get_logger(__name__)

class FocalBertFNNClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        self.alpha = [0.25, 0.75]
        self.gamma = 2

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())

                alpha = self.alpha[0] * (1 - labels.view(-1).float()) + self.alpha[1] * labels.view(-1).float()
                pt = torch.exp(-loss)
                loss = alpha * (1-pt)**self.gamma * loss
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # print('loss: ', loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    # def __init__(self, config, loss_fct):
    #     super().__init__(config)
    #     self.num_labels = config.num_labels
    #     self.config = config

    #     self.bert = BertModel(config)
    #     classifier_dropout = (
    #         config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    #     )
    #     self.dropout = nn.Dropout(classifier_dropout)
    #     self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    #     self.loss_fct = loss_fct
    #     # Initialize weights and apply final processing
    #     self.post_init()
        
    #     # super().__init__(config)
    #     # # print(config)
    #     # self.num_labels = config.num_labels
    #     # self.bert = BertModel(config)
    #     # # self.transformer = BertPreTrainedModel.from_pretrained(config)
    #     # self.score = torch.nn.Linear(config.hidden_size, self.num_labels, bias=False)
    #     # self.loss_fct = loss_fct
    #     # # Initialize weights and apply final processing
    #     # self.post_init()

    # def forward(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         token_type_ids: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         labels: Optional[torch.Tensor] = None,
    #         use_cache: Optional[bool] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, SequenceClassifierOutput]:
    #     r"""
    #     labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
    #         Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
    #         config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
    #         `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    #     """
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     outputs = self.bert(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #     # hidden_states = transformer_outputs[0]
    #     # logits = self.score(hidden_states)

    #     pooled_output = outputs[1]

    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)

    #     # if input_ids is not None:
    #     #     batch_size, sequence_length = input_ids.shape[:2]
    #     # else:
    #     #     batch_size, sequence_length = inputs_embeds.shape[:2]

    #     # assert (
    #     #         self.config.pad_token_id is not None or batch_size == 1
    #     # ), "Cannot handle batch sizes > 1 if no padding token is defined."
    #     # if self.config.pad_token_id is None:
    #     #     sequence_lengths = -1
    #     # else:
    #     #     if input_ids is not None:
    #     #         sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
    #     #     else:
    #     #         sequence_lengths = -1
    #     #         logger.warning(
    #     #             f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
    #     #             "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
    #     #         )

    #     # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    #     loss = None
    #     if labels is not None:
    #         loss_fct = BCEWithLogitsLoss()
    #         loss = loss_fct(logits, labels)
    #     #     labels = labels.to(logits.device)
    #     #     loss = self.loss_fct(pooled_logits.view(-1, ), labels.float().view(-1, ))
    #     # if not return_dict:
    #     #     output = (pooled_logits,) + transformer_outputs[1:]
    #     #     return ((loss,) + output) if loss is not None else output

    #     return SequenceClassifierOutput(
    #         loss=loss,
    #         logits=logits,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )
