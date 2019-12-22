import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

START_TAG = '<bos>'
STOP_TAG = '<eos>'

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.start_transitions.data[:] = -10000
        self.stop_transitions.data[:] = -10000

        # self.hidden = self.init_hidden()

    # def init_hidden(self):
    #     return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
    #             torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            a = self.log_sum_exp(a.unsqueeze(-1) + transitions + feat, 1)  # [batch_size, num_tags]

        return self.log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1)  # [batch_size]

    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        batch_size = tags.size()[0]
        score = torch.zeros(batch_size)

        #compute feature score
        emission = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        #compute transmision score
        tags_pairs = tags.unfold(1, 2, 1) 
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        transmission = self.transitions[indices].squeeze(0).sum(dim=-1)

        #compute start tag and stop tag score
        start_tags = torch.full((1,batch_size),self.tag_to_ix[START_TAG]).long()
        stop_tags = torch.full((1,batch_size), self.tag_to_ix[STOP_TAG]).long()
        start = self.transitions[tags[:,0], start_tags].squeeze()
        stop = self.transitions[stop_tags, tags[:,-1]].squeeze()
        
        score = emission + transmission + start + stop

        print('sentence score', score)

        return score

    def _viterbi_decode(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        v = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i]  # [batch_size, num_tags]
            v, idx = (v.unsqueeze(-1) + transitions).max(1)  # [batch_size, num_tags], [batch_size, num_tags]

            paths.append(idx)
            v = (v + feat)  # [batch_size, num_tags]

        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return (forward_score - gold_score).mean()

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq
