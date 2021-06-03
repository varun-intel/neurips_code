import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import collections

CommLayerOutput = collections.namedtuple(
    'CommLayerOutput',
    ['comms', 'comm_mask', 'gate_log_probs', 'gate_entropies', 'hidden_states']
)
PolicyOutput = collections.namedtuple(
    'PolicyOutput',
    ['actions', 'action_dists', 'hidden_states', 'comm_masks', 'gate_log_probs', 'gate_entropies']
)
ValueOutput = collections.namedtuple(
    'ValueOutput',
    ['values', 'hidden_states']
)


def get_activation_cls(activation):
    if activation == 'tanh':
        return nn.Tanh
    elif activation == 'relu':
        return nn.ReLU
    else:
        raise ValueError('Unknown activation {}.'.format(activation))


def sample_gumbel(shape_tensor, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand_like(shape_tensor)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, dim):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits)
    return F.softmax(y / temperature, dim=dim)


def onehot_from_logits(logits, dim):
    result = torch.zeros_like(logits)
    idxs = torch.argmax(logits, dim=dim, keepdims=True)
    result.scatter_(dim, idxs, torch.ones_like(logits))
    return result


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, dim=-1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y
    return y


class PolicyCommLayer(nn.Module):
    def __init__(
            self,
            num_agents, hidden_size, message_size, key_size, value_size,
            global_comm_gate, p2p_comm_gate, comm_type, comm_gate_gen, temperature,
            activation, forward_messages, use_select_comm_one,
            p2p_key_size, p2p_num_keys, p2p_gen_type
    ):
        super(PolicyCommLayer, self).__init__()
        self.comm_type = comm_type

        if self.comm_type == 'maddpg':
            global_comm_gate = p2p_comm_gate = False

        self.hidden_size = hidden_size
        self.message_size = message_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_agents = num_agents
        self.global_comm_gate = global_comm_gate
        self.p2p_comm_gate = p2p_comm_gate
        self.message_size = message_size
        self.comm_gate_gen = comm_gate_gen
        self.temperature = temperature
        self.forward_messages = forward_messages
        self.activation = activation
        self.use_select_comm_one = use_select_comm_one
        self.p2p_key_size = p2p_key_size
        self.p2p_num_keys = p2p_num_keys
        self.p2p_gen_type = p2p_gen_type

        if self.comm_type == 'maddpg':
            self.agg_dim = hidden_size
        elif self.comm_type == 'commnet':
            self.message_gen = nn.Linear(hidden_size, message_size)
            self.agg_dim = message_size
            self.initial_message = nn.Parameter(torch.zeros(message_size,), requires_grad=False)
        elif self.comm_type == 'tarmac':
            self.query_gen = nn.Linear(hidden_size, self.key_size)
            self.key_gen = nn.Linear(hidden_size, self.key_size)
            self.value_gen = nn.Linear(hidden_size, self.value_size)
            self.initial_key = nn.Parameter(torch.zeros(key_size,), requires_grad=False)
            self.initial_value = nn.Parameter(torch.zeros(value_size,), requires_grad=False)
            self.agg_dim = value_size
        elif self.comm_type == 'tarmac_sigmoid':
            self.query_gen = nn.Linear(hidden_size, self.key_size)
            self.key_gen = nn.Linear(hidden_size, self.key_size)
            self.value_gen = nn.Linear(hidden_size, self.value_size)
            self.scaling_linear = nn.Linear(1, 1)
            self.agg_dim = value_size
            self.initial_scaled_value = nn.Parameter(torch.zeros(value_size,), requires_grad=False)
        else:
            raise ValueError('Unknown comm type {}.'.format(self.comm_type))

        if self.global_comm_gate:
            self.global_comm_gate_gen = nn.Linear(self.hidden_size, 1)
        if self.p2p_comm_gate:
            if p2p_gen_type == 'context':
                if self.comm_type == 'commnet':
                    p2p_gate_input_size = self.hidden_size + self.message_size
                elif self.comm_type == 'tarmac':
                    p2p_gate_input_size = self.hidden_size + self.value_size
                elif self.comm_type == 'tarmac_sigmoid':
                    p2p_gate_input_size = self.hidden_size + self.value_size
                else:
                    raise ValueError('Unknown comm type {}.'.format(self.comm_type))
                self.p2p_comm_gate_gen = nn.Sequential(
                    nn.Linear(p2p_gate_input_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
            elif p2p_gen_type == 'p2p_keys':
                self.initial_p2p_recv_keys = nn.Parameter(torch.zeros(p2p_key_size * p2p_num_keys,), requires_grad=False)
                self.p2p_recv_key_gen = nn.Linear(hidden_size, p2p_key_size * p2p_num_keys)
                self.p2p_send_key_gen = nn.Linear(hidden_size, p2p_key_size)
                self.p2p_comm_gate_gen = nn.Sequential(
                    nn.Linear(p2p_num_keys, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
            elif p2p_gen_type == 'p2p_probs':
                self.initial_p2p_recv_probs = nn.Parameter(torch.ones(p2p_num_keys,), requires_grad=False)
                self.p2p_recv_prob_gen = nn.Linear(hidden_size, p2p_num_keys)
                self.p2p_send_prob_gen = nn.Linear(hidden_size, p2p_num_keys)
            elif p2p_gen_type == 'p2p_scores':
                self.initial_p2p_recv_scores = nn.Parameter(torch.zeros(p2p_num_keys,), requires_grad=False)
                self.p2p_recv_score_gen = nn.Linear(hidden_size, p2p_num_keys)
                self.p2p_send_prob_gen = nn.Linear(hidden_size, p2p_num_keys)
                self.p2p_comm_gate_gen = nn.Linear(1, 1)
            elif p2p_gen_type == 'p2p_obs':
                self.p2p_comm_gate_gen = nn.Linear(hidden_size, 1)
            elif p2p_gen_type == 'p2p_obs_deep':
                self.p2p_comm_gate_gen = nn.Sequential(
                    nn.Linear(3 * hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
            else:
                raise ValueError('Unknown p2p gen type {}.'.format(p2p_gen_type))
        self.output = nn.Linear(self.agg_dim, hidden_size)

    def reset_hidden_states(self, num_envs, device):
        hidden_states = {}
        if self.comm_type == 'maddpg':
            pass
        elif self.comm_type == 'commnet':
            messages = self.initial_message[None, None, None].expand(num_envs, self.num_agents, self.num_agents, self.message_size)
            hidden_states['messages'] = messages
        elif self.comm_type == 'tarmac':
            keys = self.initial_key[None, None, None].expand(num_envs, self.num_agents, self.num_agents, self.key_size)
            values = self.initial_value[None, None, None].expand(num_envs, self.num_agents, self.num_agents, self.value_size)
            hidden_states['keys'] = keys
            hidden_states['values'] = values
        elif self.comm_type == 'tarmac_sigmoid':
            scaled_values = self.initial_scaled_value[None, None, None].expand(num_envs, self.num_agents, self.num_agents, self.value_size)
            hidden_states['scaled_values'] = scaled_values
        else:
            raise ValueError('Unknown comm type {}.'.format(self.comm_type))

        if self.p2p_comm_gate and self.p2p_gen_type == 'p2p_keys':
            p2p_recv_keys = self.initial_p2p_recv_keys[None, None, None].expand(num_envs, self.num_agents, self.num_agents, self.p2p_key_size * self.p2p_num_keys)
            hidden_states['p2p_recv_keys'] = p2p_recv_keys

        if self.p2p_comm_gate and self.p2p_gen_type == 'p2p_probs':
            p2p_recv_probs = self.initial_p2p_recv_probs[None, None, None].expand(num_envs, self.num_agents, self.num_agents, self.p2p_num_keys)
            hidden_states['p2p_recv_probs'] = p2p_recv_probs

        if self.p2p_comm_gate and self.p2p_gen_type == 'p2p_scores':
            p2p_recv_scores = self.initial_p2p_recv_scores[None, None, None].expand(num_envs, self.num_agents, self.num_agents, self.p2p_num_keys)
            hidden_states['p2p_recv_scores'] = p2p_recv_scores

        return hidden_states

    def detach_hidden_states(self, hidden_states):
        return {k: v.detach() for k, v in hidden_states.items()}

    def make_gate_dist(self, x, stochastic, is_train):
        dist = torch.distributions.Bernoulli(logits=x)
        if stochastic:
            if self.comm_gate_gen == 'sample_softmax':
                gate = dist.sample()
            elif self.comm_gate_gen == 'softmax_st':
                gate = (x >= 0).float()
                probs = torch.sigmoid(x)
                gate = (gate - probs).detach() + probs
            elif self.comm_gate_gen == 'gumbel_softmax':
                logits = torch.stack([x, torch.zeros_like(x)], dim=-1)
                gate = gumbel_softmax(logits, temperature=self.temperature, hard=True)[..., 0]
            elif self.comm_gate_gen == 'gumbel_softmax_soft':
                logits = torch.stack([x, torch.zeros_like(x)], dim=-1)
                if is_train:
                    gate = gumbel_softmax(logits, temperature=self.temperature, hard=False)[..., 0]
                else:
                    gate = gumbel_softmax(logits, temperature=self.temperature, hard=True)[..., 0]
            else:
                raise ValueError('Unknown gen method {}.'.format(self.comm_gate_gen))
        else:
            gate = (x >= 0).float()
        return dist, gate

    def forward_global_gate(self, x, comm_penalty_masks, stochastic, is_train):
        x = self.global_comm_gate_gen(x).squeeze(-1)
        return self.make_gate_dist(x=x, stochastic=stochastic, is_train=is_train)

    def forward_p2p_gate(self, x, obs, p2p_obs, comm_penalty_masks, hidden_states, stochastic, is_train):
        if self.p2p_gen_type == 'context':
            bz, na, _ = x.shape
            x = x[:, :, None, :].expand(bz, na, na, -1)
            if self.comm_type == 'commnet':
                x = torch.cat([x, hidden_states['messages'].transpose(1, 2)], dim=-1)
            elif self.comm_type == 'tarmac':
                x = torch.cat([x, hidden_states['values'].transpose(1, 2)], dim=-1)
            elif self.comm_type == 'tarmac_sigmoid':
                x = torch.cat([x, hidden_states['scaled_values'].transpose(1, 2)], dim=-1)
            else:
                raise ValueError('Unknown comm type {}.'.format(self.comm_type))

            x = self.p2p_comm_gate_gen(x).squeeze(-1)
            return self.make_gate_dist(x=x, stochastic=stochastic, is_train=is_train)
        elif self.p2p_gen_type == 'p2p_keys':
            p2p_recv_keys = hidden_states['p2p_recv_keys'].transpose(1, 2)
            p2p_recv_keys = p2p_recv_keys.view(*p2p_recv_keys.shape[:-1], self.p2p_num_keys, self.p2p_key_size)
            p2p_send_keys = self.p2p_send_key_gen(x)[:, :, None, None, :]
            p2p_scores = (p2p_recv_keys * p2p_send_keys).sum(-1)
            p2p_scores = self.p2p_comm_gate_gen(p2p_scores).squeeze(-1)
            return self.make_gate_dist(x=p2p_scores, stochastic=stochastic, is_train=is_train)
        elif self.p2p_gen_type == 'p2p_probs':
            p2p_recv_probs = hidden_states['p2p_recv_probs'].transpose(1, 2)
            p2p_send_probs = torch.softmax(self.p2p_send_prob_gen(x)[:, :, None, :], dim=-1)
            p2p_scores = (p2p_recv_probs * p2p_send_probs).sum(-1)
            p2p_scores = torch.log(p2p_scores)
            return self.make_gate_dist(x=p2p_scores, stochastic=stochastic, is_train=is_train)
        elif self.p2p_gen_type == 'p2p_scores':
            p2p_recv_scores = hidden_states['p2p_recv_scores'].transpose(1, 2)
            p2p_send_probs = torch.softmax(self.p2p_send_prob_gen(x)[:, :, None, :], dim=-1)
            p2p_scores = (p2p_recv_scores * p2p_send_probs).sum(-1)
            p2p_scores = self.p2p_comm_gate_gen(p2p_scores[..., None]).squeeze(-1)
            return self.make_gate_dist(x=p2p_scores, stochastic=stochastic, is_train=is_train)
        elif self.p2p_gen_type == 'p2p_obs':
            p2p_scores = self.p2p_comm_gate_gen(p2p_obs).squeeze(-1)
            return self.make_gate_dist(x=p2p_scores, stochastic=stochastic, is_train=is_train)
        elif self.p2p_gen_type == 'p2p_obs_deep':
            p2p_inputs = [
                x[:, :, None, :].expand(-1, -1, self.num_agents, -1),
                obs[:, :, None, :].expand(-1, -1, self.num_agents, -1),
                p2p_obs
            ]
            p2p_scores = self.p2p_comm_gate_gen(torch.cat(p2p_inputs, dim=-1)).squeeze(-1)
            return self.make_gate_dist(x=p2p_scores, stochastic=stochastic, is_train=is_train)
        else:
            raise ValueError('Unknown p2p gen type {}.'.format(self.p2p_gen_type))

    def maybe_reset_hidden_states(self, hidden_states, prev_step_masks):
        fresh_hidden_states = self.reset_hidden_states(prev_step_masks.size(0), prev_step_masks.device)
        prev_step_masks = prev_step_masks[:, None, None, None]

        new_hidden_states = {}
        for k, v in hidden_states.items():
            new_hidden_states[k] = prev_step_masks * hidden_states[k] + (1 - prev_step_masks) * fresh_hidden_states[k]
        return new_hidden_states

    def make_forward_messages(self, messages, prev_step_masks, forward_mask, hidden_states):
        forward_mask = forward_mask[..., None] * prev_step_masks[:, None, None, None]

        if not self.forward_messages:
            hidden_states = self.reset_hidden_states(prev_step_masks.size(0), prev_step_masks.device)

        new_messages = {}
        for k, v in messages.items():
            new_messages[k] = forward_mask * v + (1 - forward_mask) * hidden_states[k]
        return new_messages

    def forward(self, x, obs, p2p_obs, comm_penalty_masks, prev_step_masks, alive_masks, stochastic, is_train, hidden_states, comm_mask):
        x = x * prev_step_masks[:, None, None]
        hidden_states = self.maybe_reset_hidden_states(hidden_states, prev_step_masks)
        new_hidden_states = {}

        # Generate global communication gates
        bz, device = x.size(0), x.device
        if self.global_comm_gate:
            global_gate_dist, global_gate = self.forward_global_gate(
                x=x,
                comm_penalty_masks=comm_penalty_masks,
                stochastic=stochastic,
                is_train=is_train
            )
            global_gate_log_probs = global_gate_dist.log_prob(global_gate)
            global_gate_entropies = global_gate_dist.entropy()
        else:
            global_gate = torch.ones((bz, self.num_agents), dtype=torch.float32, device=device)
            global_gate_log_probs = torch.zeros((bz, self.num_agents), dtype=torch.float32, device=device)
            global_gate_entropies = torch.zeros((bz, self.num_agents), dtype=torch.float32, device=device)

        if self.p2p_comm_gate:
            p2p_gate_dist, p2p_gate = self.forward_p2p_gate(
                x=x,
                obs=obs,
                p2p_obs=p2p_obs,
                hidden_states=hidden_states,
                comm_penalty_masks=comm_penalty_masks,
                stochastic=stochastic,
                is_train=is_train
            )
            p2p_gate_log_probs = p2p_gate_dist.log_prob(p2p_gate).sum(-1)
            p2p_gate_entropies = p2p_gate_dist.entropy().sum(-1)
        else:
            p2p_gate = torch.ones((bz, self.num_agents, self.num_agents), dtype=torch.float32, device=device)
            p2p_gate_log_probs = torch.zeros((bz, self.num_agents), dtype=torch.float32, device=device)
            p2p_gate_entropies = torch.zeros((bz, self.num_agents), dtype=torch.float32, device=device)

        gate_log_probs = global_gate_log_probs + p2p_gate_log_probs
        gate_entropies = global_gate_entropies + p2p_gate_entropies
        forward_mask = global_gate[:, :, None] * p2p_gate

        if self.use_select_comm_one:
            forward_mask = comm_penalty_masks[:, None, None] * forward_mask + (1 - comm_penalty_masks[:, None, None]) * torch.ones_like(forward_mask)

        if comm_mask is None:
            # comm_mask shape (bz, na: sender, na: receiver).
            comm_mask = torch.ones((bz, self.num_agents, self.num_agents), dtype=torch.float32, device=device)

        comm_mask = comm_mask * prev_step_masks[:, None, None]
        comm_mask = comm_mask * alive_masks[:, :, None].float()
        comm_mask = comm_mask * alive_masks[:, None, :].float()

        if self.p2p_comm_gate and self.p2p_gen_type == 'p2p_keys':
            p2p_recv_keys = self.p2p_recv_key_gen(x)
            p2p_recv_keys = p2p_recv_keys[:, :, None, :]
            new_hidden_states['p2p_recv_keys'] = self.make_forward_messages({'p2p_recv_keys': p2p_recv_keys}, prev_step_masks, forward_mask, hidden_states)['p2p_recv_keys']

        if self.p2p_comm_gate and self.p2p_gen_type == 'p2p_probs':
            p2p_recv_probs = torch.sigmoid(self.p2p_recv_prob_gen(x))
            p2p_recv_probs = p2p_recv_probs[:, :, None, :]
            new_hidden_states['p2p_recv_probs'] = self.make_forward_messages({'p2p_recv_probs': p2p_recv_probs}, prev_step_masks, forward_mask, hidden_states)['p2p_recv_probs']

        if self.p2p_comm_gate and self.p2p_gen_type == 'p2p_scores':
            p2p_recv_scores = self.p2p_recv_score_gen(x)
            p2p_recv_scores = p2p_recv_scores[:, :, None, :]
            new_hidden_states['p2p_recv_scores'] = self.make_forward_messages({'p2p_recv_scores': p2p_recv_scores}, prev_step_masks, forward_mask, hidden_states)['p2p_recv_scores']

        if self.comm_type == 'maddpg':
            comms = x
        elif self.comm_type == 'commnet':
            messages = self.message_gen(x)  # x has shape (bz, na, h)
            messages = messages[:, :, None, :]  # expand to (bz, na:sender, 1:receiver, h)

            # forward previous messages
            new_messages = {'messages': messages}
            new_messages = self.make_forward_messages(new_messages, prev_step_masks, forward_mask, hidden_states)
            messages = new_hidden_states['messages'] = new_messages['messages']

            # apply comm mask and average
            messages = messages * comm_mask[..., None]
            num_incoming = comm_mask.sum(1, keepdims=True)[..., None]
            num_incoming[num_incoming == 0] = 1
            messages = messages / num_incoming
            messages = messages.sum(1)

            comms = messages
        elif self.comm_type == 'tarmac':
            keys = self.key_gen(x)[:, :, None, :]
            values = self.value_gen(x)[:, :, None, :]

            # forward previous messages
            new_messages = {'keys': keys, 'values': values}
            new_messages = self.make_forward_messages(new_messages, prev_step_masks, forward_mask, hidden_states)
            keys = new_hidden_states['keys'] = new_messages['keys']
            values = new_hidden_states['values'] = new_messages['values']

            queries = self.query_gen(x)[:, None, :, :]
            scores = (queries * keys).sum(-1) / np.sqrt(self.key_size)
            scores = scores.masked_fill(comm_mask == 0, -1e15)
            scores = F.softmax(scores, dim=1)
            comms = (values * scores[..., None]).sum(1)
        elif self.comm_type == 'tarmac_sigmoid':
            keys = self.key_gen(x)[:, :, None, :]
            values = self.value_gen(x)[:, :, None, :]
            queries = self.query_gen(x)[:, None, :, :]
            scores = (queries * keys).sum(-1) / np.sqrt(self.key_size)
            scores = self.scaling_linear(scores[..., None]).squeeze(-1)
            scores = torch.sigmoid(scores)
            scaled_values = scores[..., None] * values

            new_messages = {'scaled_values': scaled_values}
            new_messages = self.make_forward_messages(new_messages, prev_step_masks, forward_mask, hidden_states)
            scaled_values = new_hidden_states['scaled_values'] = new_messages['scaled_values']
            scaled_values = scaled_values * comm_mask[..., None]
            comms = scaled_values.sum(1)
        else:
            raise ValueError('Unknown comm type {}.'.format(self.comm_type))

        comms = self.output(comms)
        comms = comms * prev_step_masks[:, None, None]

        # apply global gate to comm mask to track communication that actually happened
        comm_mask = comm_mask * forward_mask
        self_mask = torch.ones_like(comm_mask)
        self_mask[:, range(self.num_agents), range(self.num_agents)] = 0
        comm_mask = comm_mask * self_mask

        return CommLayerOutput(
            comms=comms,
            comm_mask=comm_mask,
            gate_log_probs=gate_log_probs,
            gate_entropies=gate_entropies,
            hidden_states=new_hidden_states
        )


class PolicyNetwork(nn.Module):
    def __init__(
            self,
            action_space, obs_size, p2p_obs_size,
            hidden_size, message_size, key_size, value_size,
            num_agents, comm_type,
            num_comm_rounds, forward_messages,
            global_comm_gate, p2p_comm_gate, use_select_comm_one,
            comm_gate_gen, temperature,
            activation, p2p_key_size, p2p_num_keys, p2p_gen_type
    ):
        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        self.action_is_discrete = isinstance(self.action_space, gym.spaces.Discrete)
        if self.action_is_discrete:
            self.action_dim = self.action_space.n
        else:
            self.action_dim = np.product(self.action_space.shape)
        self.obs_size = obs_size
        self.p2p_obs_size = p2p_obs_size
        self.key_size = key_size
        self.message_size = message_size
        self.value_size = value_size
        self.hidden_size = hidden_size
        self.num_agents = num_agents
        self.comm_type = comm_type
        self.global_comm_gate = global_comm_gate
        self.p2p_comm_gate = p2p_comm_gate
        self.use_select_comm_one = use_select_comm_one
        self.num_comm_rounds = num_comm_rounds
        self.comm_gate_gen = comm_gate_gen
        self.temperature = temperature
        self.forward_messages = forward_messages
        self.activation = activation

        self.obs_embed = nn.Linear(self.obs_size + 1, hidden_size)
        self.p2p_obs_embed = nn.Sequential(nn.Linear(self.obs_size + 1 + self.p2p_obs_size, hidden_size), nn.Tanh())
        self.rnn = nn.LSTMCell(2 * hidden_size, hidden_size)

        self.comm_layer = PolicyCommLayer(
            num_agents=num_agents,
            hidden_size=hidden_size,
            message_size=message_size,
            global_comm_gate=global_comm_gate,
            p2p_comm_gate=p2p_comm_gate,
            comm_type=comm_type,
            comm_gate_gen=comm_gate_gen,
            temperature=temperature,
            key_size=key_size,
            value_size=value_size,
            activation=activation,
            forward_messages=forward_messages,
            use_select_comm_one=use_select_comm_one,
            p2p_key_size=p2p_key_size,
            p2p_num_keys=p2p_num_keys,
            p2p_gen_type=p2p_gen_type
        )
        if self.action_is_discrete:
            self.action_output = nn.Linear(hidden_size, self.action_dim)
        else:
            self.action_output = nn.Linear(hidden_size, 2 * self.action_dim)

    def reset_hidden_states(self, num_envs, device):
        hidden_states = {}
        hidden_states['rnn'] = (
            torch.zeros((num_envs, self.num_agents, self.hidden_size), dtype=torch.float32, device=device),
            torch.zeros((num_envs, self.num_agents, self.hidden_size), dtype=torch.float32, device=device)
        )
        hidden_states['comms'] = self.comm_layer.reset_hidden_states(num_envs, device)
        return hidden_states

    def detach_hidden_states(self, hidden_states):
        new_hidden_states = {}
        new_hidden_states['rnn'] = (hidden_states['rnn'][0].detach(), hidden_states['rnn'][1].detach())
        new_hidden_states['comms'] = self.comm_layer.detach_hidden_states(hidden_states['comms'])
        return new_hidden_states

    def forward_rnn_step(self, x, hidden_states, masks):
        h, c = hidden_states
        h, c = h * masks[:, None, None], c * masks[:, None, None]

        bz, na = x.size(0), x.size(1)
        x = x.view(bz * na, x.size(-1))
        h = h.view(bz * na, h.size(-1))
        c = c.view(bz * na, c.size(-1))

        h, c = self.rnn(x, (h, c))

        h = h.view(bz, na, h.size(-1))
        c = c.view(bz, na, c.size(-1))

        return h, c

    def act(self, obs, p2p_obs, comm_penalty_masks, hidden_states, prev_step_masks, alive_masks, stochastic, is_train):
        hidden_states = {'rnn': hidden_states['rnn'], 'comms': hidden_states['comms']}
        comm_outputs = []

        # Embed observation
        bz, na, _ = obs.shape
        obs = torch.cat([obs, comm_penalty_masks[:, None, None].expand(bz, na, 1)], dim=-1)
        p2p_obs = torch.cat([obs[:, :, None, :].expand(bz, na, na, -1), p2p_obs], dim=-1)

        # Embed obs, p2p obs
        obs, p2p_obs = self.obs_embed(obs), self.p2p_obs_embed(p2p_obs)

        # Run message rounds
        for comm_round in range(self.num_comm_rounds):
            comm_round_output = self.comm_layer(
                x=hidden_states['rnn'][0],
                obs=obs,
                p2p_obs=p2p_obs,
                comm_mask=None,
                comm_penalty_masks=comm_penalty_masks,
                alive_masks=alive_masks,
                stochastic=stochastic,
                is_train=is_train,
                prev_step_masks=prev_step_masks,
                hidden_states=hidden_states['comms']
            )
            comm_outputs.append(comm_round_output)
            hidden_states['comms'] = comm_round_output.hidden_states

            hidden_states['rnn'] = self.forward_rnn_step(
                x=torch.cat([obs, comm_round_output.comms], dim=-1),
                hidden_states=hidden_states['rnn'],
                masks=prev_step_masks
            )

            # We can now use the previous hidden states
            prev_step_masks = torch.ones_like(prev_step_masks)

        action_out = self.action_output(hidden_states['rnn'][0])
        if self.action_is_discrete:
            action_dist = torch.distributions.Categorical(logits=action_out)
            if stochastic:
                actions = action_dist.sample()
            else:
                actions = action_out.argmax(-1)
        else:
            action_dist = torch.distributions.Normal(
                action_out[..., :self.action_dim],
                action_out[..., self.action_dim:].exp()
            )
            if self.stochastic:
                actions = action_dist.sample()
            else:
                actions = action_out[..., :self.action_dim]

        return PolicyOutput(
            actions=actions,
            action_dists=action_dist,
            hidden_states=hidden_states,
            comm_masks=torch.stack([comm_round_output.comm_mask for comm_round_output in comm_outputs], dim=-1),
            gate_log_probs=torch.stack([comm_round_output.gate_log_probs for comm_round_output in comm_outputs], dim=-1),
            gate_entropies=torch.stack([comm_round_output.gate_entropies for comm_round_output in comm_outputs], dim=-1)
        )


class ValueNetworkInner(nn.Module):
    def __init__(
            self,
            obs_size, hidden_size,
            message_size, key_size, value_size,
            team_num_agents, team_idx,
            num_comm_rounds, comm_type,
            activation, shared_comm_penalty_ratio
    ):
        super(ValueNetworkInner, self).__init__()
        self.obs_size = obs_size
        self.team_num_agents = team_num_agents
        self.team_idx = team_idx
        self.my_team_num_agents = team_num_agents[team_idx]
        self.num_comm_rounds = num_comm_rounds
        self.hidden_size = hidden_size
        self.comm_type = comm_type
        self.activation = activation
        self.shared_comm_penalty_ratio = shared_comm_penalty_ratio

        self.obs_embed = nn.Linear(self.obs_size + 1, hidden_size)
        self.rnn = nn.LSTMCell(3 * hidden_size + 2, hidden_size)
        self.comm_layer = PolicyCommLayer(
            num_agents=self.my_team_num_agents,
            hidden_size=hidden_size,
            message_size=message_size,
            key_size=key_size,
            value_size=value_size,
            global_comm_gate=False,
            p2p_comm_gate=False,
            comm_type=self.comm_type,
            temperature=1,
            comm_gate_gen="sample_softmax",
            activation=activation,
            forward_messages=False,
            use_select_comm_one=False,
            p2p_key_size=0,
            p2p_num_keys=0,
            p2p_gen_type=''
        )
        self.simulate_comm_layer = PolicyCommLayer(
            num_agents=self.my_team_num_agents,
            hidden_size=hidden_size,
            message_size=message_size,
            key_size=key_size,
            value_size=value_size,
            global_comm_gate=False,
            p2p_comm_gate=False,
            comm_type=self.comm_type,
            temperature=1,
            comm_gate_gen="sample_softmax",
            activation=activation,
            forward_messages=False,
            use_select_comm_one=False,
            p2p_key_size=0,
            p2p_num_keys=0,
            p2p_gen_type=''
        )
        self.value_gen = nn.Linear(hidden_size, 1)

    def reset_hidden_states(self, num_envs, device):
        hidden_states = {}
        hidden_states['rnn'] = (
            torch.zeros((num_envs, self.my_team_num_agents, self.hidden_size), dtype=torch.float32, device=device),
            torch.zeros((num_envs, self.my_team_num_agents, self.hidden_size), dtype=torch.float32, device=device)
        )
        hidden_states['comms'] = self.comm_layer.reset_hidden_states(num_envs, device)
        hidden_states['sim_comms'] = self.simulate_comm_layer.reset_hidden_states(num_envs, device)
        return hidden_states

    def detach_hidden_states(self, hidden_states):
        new_hidden_states = {}
        new_hidden_states['rnn'] = (hidden_states['rnn'][0].detach(), hidden_states['rnn'][1].detach())
        new_hidden_states['comms'] = self.comm_layer.detach_hidden_states(hidden_states['comms'])
        new_hidden_states['sim_comms'] = self.simulate_comm_layer.detach_hidden_states(hidden_states['sim_comms'])
        return new_hidden_states

    def forward_rnn_step(self, x, hidden_states, masks):
        bz, na = x.size(0), x.size(1)

        h, c = hidden_states
        h, c = h * masks[:, None, None], c * masks[:, None, None]
        x, h, c = x.view(bz * na, -1), h.view(bz * na, -1), c.view(bz * na, -1)

        h, c = self.rnn(x, (h, c))
        h = h.view(bz, na, -1)
        c = c.view(bz, na, -1)

        return h, c

    def forward(self, obs, comm_masks, comm_penalty_masks, hidden_states, prev_step_masks, alive_masks):
        hidden_states = {'rnn': hidden_states['rnn'], 'comms': hidden_states['comms'], 'sim_comms': hidden_states['sim_comms']}

        obs = obs[self.team_idx]
        comm_masks = comm_masks[self.team_idx]
        comm_penalty_masks = comm_penalty_masks[self.team_idx]
        alive_masks = alive_masks[self.team_idx]

        bz, na, _ = obs.shape
        obs = torch.cat([obs, comm_penalty_masks[:, None, None].expand(bz, na, 1)], dim=-1)
        obs = self.obs_embed(obs)

        for comm_round in range(self.num_comm_rounds):
            comm_round_output = self.comm_layer(
                x=hidden_states['rnn'][0],
                obs=None,
                p2p_obs=None,
                alive_masks=alive_masks,
                stochastic=False,
                is_train=False,
                prev_step_masks=prev_step_masks,
                hidden_states=hidden_states['comms'],
                comm_mask=None,
                comm_penalty_masks=None
            )
            hidden_states['comms'] = comm_round_output.hidden_states

            if comm_masks.size(-1) == 0:
                sim_comms = torch.zeros_like(comm_round_output.comms)
                round_num_comms = torch.zeros((bz, na), dtype=comm_masks.dtype, device=comm_masks.device)
            else:
                round_comm_mask = comm_masks[..., min(comm_round, comm_masks.size(-1) - 1)]
                self_mask = torch.zeros_like(round_comm_mask)
                self_mask[:, range(self.my_team_num_agents), range(self.my_team_num_agents)] = 1
                round_comm_mask = self_mask + (1 - self_mask) * round_comm_mask

                sim_comm_round_output = self.simulate_comm_layer(
                    x=hidden_states['rnn'][0],
                    obs=None,
                    p2p_obs=None,
                    alive_masks=alive_masks,
                    stochastic=False,
                    is_train=False,
                    prev_step_masks=prev_step_masks,
                    hidden_states=hidden_states['sim_comms'],
                    comm_mask=round_comm_mask,
                    comm_penalty_masks=None
                )
                hidden_states['sim_comms'] = sim_comm_round_output.hidden_states
                sim_comms = sim_comm_round_output.comms

            if comm_round < comm_masks.size(-1):
                round_num_comms = comm_masks[..., comm_round].sum(-1)
            else:
                round_num_comms = torch.zeros((bz, na), dtype=comm_masks.dtype, device=comm_masks.device)
            shared_round_num_comms = self.shared_comm_penalty_ratio * round_num_comms.mean(-1, keepdims=True) + (1 - self.shared_comm_penalty_ratio) * round_num_comms
            shared_round_num_comms = comm_penalty_masks[:, None] * shared_round_num_comms

            hidden_states['rnn'] = self.forward_rnn_step(
                x=torch.cat([obs, comm_round_output.comms, sim_comms, round_num_comms[..., None], shared_round_num_comms[..., None]], dim=-1),
                hidden_states=hidden_states['rnn'],
                masks=prev_step_masks
            )

            # We can now use the previous hidden states
            prev_step_masks = torch.ones_like(prev_step_masks)

        values = self.value_gen(hidden_states['rnn'][0]).squeeze(-1)
        values = torch.split(values, self.team_num_agents, dim=1)
        values = values[self.team_idx]

        return ValueOutput(
            values=values,
            hidden_states=hidden_states
        )


class ValueNetwork(nn.Module):
    def __init__(
            self,
            obs_size, hidden_size,
            message_size, key_size, value_size,
            team_num_agents, team_idx,
            num_comm_rounds, comm_type,
            activation, shared_comm_penalty_ratio
    ):
        super(ValueNetwork, self).__init__()
        self.num_comm_rounds = num_comm_rounds
        self.team_num_agents = team_num_agents
        inner_networks = []
        for i in range(num_comm_rounds + 1):
            inner_net = ValueNetworkInner(
                obs_size=obs_size, hidden_size=hidden_size,
                message_size=message_size, key_size=key_size, value_size=value_size,
                team_num_agents=team_num_agents, team_idx=team_idx,
                num_comm_rounds=num_comm_rounds, comm_type=comm_type,
                activation=activation, shared_comm_penalty_ratio=shared_comm_penalty_ratio
            )
            inner_networks.append(inner_net)
        self.inner_networks = nn.ModuleList(inner_networks)

    def reset_hidden_states(self, num_envs, device):
        return [inner_net.reset_hidden_states(num_envs=num_envs, device=device) for inner_net in self.inner_networks]

    def detach_hidden_states(self, hidden_states):
        return [inner_net.detach_hidden_states(h) for inner_net, h in zip(self.inner_networks, hidden_states)]

    def forward(self, obs, comm_penalty_masks, hidden_states, prev_step_masks, alive_masks, comm_masks):
        comm_masks = [team_comm_masks.detach() for team_comm_masks in comm_masks]
        values = []
        new_hidden_states = []
        for i in range(self.num_comm_rounds + 1):
            output_i = self.inner_networks[i](
                obs=obs,
                comm_penalty_masks=comm_penalty_masks,
                comm_masks=[team_comm_masks[..., :i] for team_comm_masks in comm_masks],
                hidden_states=hidden_states[i],
                prev_step_masks=prev_step_masks,
                alive_masks=alive_masks
            )
            values.append(output_i.values)
            new_hidden_states.append(output_i.hidden_states)
        values = torch.stack(values, dim=-1)
        return ValueOutput(
            values=values,
            hidden_states=new_hidden_states
        )
