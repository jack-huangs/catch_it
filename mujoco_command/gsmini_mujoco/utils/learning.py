import random
import warnings
from collections import deque, namedtuple
from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done", "episode")
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        out_scale: float = 1.0,
        hidden_size: int = 256,
        hidden_activation_func: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        output_activation_func: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        """
        Multi-Layer Perceptron (MLP) neural network.

        This class implements a multi-layer perceptron (MLP) with two hidden layers and
        customizable activation functions. It scales the output by a specified factor.

        Parameters
        ----------
        - input_dim : int
            Dimension of the input layer.
        - output_dim : int
            Dimension of the output layer.
        - out_scale : float, optional
            Scaling factor for the output. Defaults to 1.0.
        - hidden_size : int, optional
            Number of neurons in each hidden layer. Defaults to 256.
        - hidden_activation_func : Callable[[torch.Tensor], torch.Tensor], optional
            Activation function for the hidden layers. Defaults to `torch.relu`.
        - output_activation_func : Callable[[torch.Tensor], torch.Tensor], optional
            Activation function for the output layer. Defaults to `torch.tanh`.

        Returns
        ----------
        - None
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_dim)
        self.hidden_activate = hidden_activation_func
        self.output_activate = output_activation_func
        self.out_scale = (
            out_scale
            if isinstance(out_scale, torch.Tensor)
            else torch.tensor(out_scale).to(device)
        )

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of the MLP.

        Computes the output of the MLP for a given input state.

        Parameters
        ----------
        state : torch.FloatTensor
            Input state tensor.

        Returns
        ----------
        torch.FloatTensor
            Output tensor after passing through the network.
        """
        if isinstance(state, list):
            state = torch.FloatTensor(state)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(device)
        a = self.hidden_activate(self.l1(state)).to(device)
        a = self.hidden_activate(self.l2(a)).to(device)
        res = self.out_scale * self.output_activate(self.l3(a)).to(device)
        return res

    def get_input_dim(self) -> int:
        """
        Get the input dimension of the MLP.

        Returns
        ----------
        int
            Input dimension.
        """
        return self.input_dim

    def get_output_dim(self) -> int:
        """
        Get the output dimension of the MLP.

        Returns
        ----------
        int
            Output dimension.
        """
        return self.output_dim


class ReplayMemory:
    def __init__(self, capacity: int, normalization: bool = False) -> None:
        """
        Replay memory buffer for storing transitions.

        This class maintains a buffer of past transitions to be used for training.
        Transitions are stored in a deque with a fixed maximum capacity.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store in the buffer.
        normalization : bool, optional
            Whether to normalize the transitions. Defaults to False.

        Returns
        ----------
        None
        """
        self.states = None
        self.actions = None
        self.next_states = None
        self.rewards = None
        self.dones = None
        self.episodes = None
        self.replay_buffer = None
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.normalize = normalization
        self.ordered_episodes = []

    def __eq__(self, other: object) -> bool:
        """
        Check if two ReplayMemory objects are equal.

        Compares the contents of two ReplayMemory objects to determine equality.

        Parameters
        ----------
        other : object
            Another ReplayMemory object to compare with.

        Returns
        ----------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, ReplayMemory):
            return False

        replay_buffer = Transition(*zip(*self.memory))
        other_replay_buffer = Transition(*zip(*other.memory))
        ans = isinstance(replay_buffer.state[0], type(other_replay_buffer.state[0]))
        ans = (
            ans
            and (
                np.asarray(replay_buffer.state) == np.asarray(other_replay_buffer.state)
            ).all()
        )
        ans = (
            ans
            and (
                np.asarray(replay_buffer.action)
                == np.asarray(other_replay_buffer.action)
            ).all()
        )
        ans = (
            ans
            and (
                np.asarray(replay_buffer.next_state)
                == np.asarray(other_replay_buffer.next_state)
            ).all()
        )
        ans = (
            ans
            and (
                np.asarray(replay_buffer.reward)
                == np.asarray(other_replay_buffer.reward)
            ).all()
        )
        ans = (
            ans
            and (
                np.asarray(replay_buffer.done) == np.asarray(other_replay_buffer.done)
            ).all()
        )
        return ans

    def push(self, *args: Tuple) -> None:
        """
        Save a transition in the memory.

        Adds a new transition to the replay memory.

        Parameters
        ----------
        *args : Tuple
            Transition tuple (state, action, next_state, reward, done, episode).

        Returns
        ----------
        None
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        """
        Sample a batch of transitions from the memory.

        Randomly samples a specified number of transitions from the memory and returns them as a batch.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        ----------
        Transition
            Batch of sampled transitions.
        """
        warnings.simplefilter("ignore", category=FutureWarning)
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.from_numpy(np.array(batch.state, dtype=np.float32)).to(device)
        actions = torch.from_numpy(np.array(batch.action, dtype=np.float32)).to(device)
        next_states = torch.from_numpy(np.array(batch.next_state, dtype=np.float32)).to(
            device
        )
        rewards = torch.reshape(torch.FloatTensor(batch.reward).to(device), (-1, 1))
        dones = torch.reshape(torch.FloatTensor(batch.done).to(device), (-1, 1))
        episodes = torch.reshape(torch.FloatTensor(batch.episode).to(device), (-1, 1))
        return Transition(states, actions, next_states, rewards, dones, episodes)

    def __len__(self) -> int:
        """
        Get the number of transitions stored in the memory.

        Returns
        ----------
        int
            Number of transitions.
        """
        return len(self.memory)

    def save(self, filename: str = "replay_buffer") -> None:
        """
        Save the replay memory to files.

        Exports the replay memory to numpy binary files with specified base filename.

        Parameters
        ----------
        filename : str, optional
            Base filename for saving the memory. Defaults to "replay_buffer".

        Returns
        ----------
        None
        """
        replay_buffer = Transition(*zip(*self.memory))
        np.save(filename + "_states", np.array(replay_buffer.state))
        np.save(filename + "_actions", np.array(replay_buffer.action))
        np.save(filename + "_next_states", np.array(replay_buffer.next_state))
        np.save(filename + "_rewards", np.array(replay_buffer.reward))
        np.save(filename + "_dones", np.array(replay_buffer.done))
        np.save(filename + "_episodes", np.array(replay_buffer.episode))

    def savetxt(self, filename: str = "replay_buffer", delimiter: str = " ") -> None:
        """
        Save the replay memory to text files.

        Exports the replay memory to text files with specified base filename and delimiter.

        Parameters
        ----------
        filename : str, optional
            Base filename for saving the memory. Defaults to "replay_buffer".
        delimiter : str, optional
            Delimiter for the text files. Defaults to " ".

        Returns
        ----------
        None
        """
        replay_buffer = Transition(*zip(*self.memory))
        np.savetxt(
            filename + "_states", np.array(replay_buffer.state), delimiter=delimiter
        )
        np.savetxt(
            filename + "_actions", np.array(replay_buffer.action), delimiter=delimiter
        )
        np.savetxt(
            filename + "_next_states",
            np.array(replay_buffer.next_state),
            delimiter=delimiter,
        )
        np.savetxt(
            filename + "_rewards", np.array(replay_buffer.reward), delimiter=delimiter
        )
        np.savetxt(
            filename + "_dones", np.array(replay_buffer.done), delimiter=delimiter
        )
        np.savetxt(
            filename + "_episodes", np.array(replay_buffer.episode), delimiter=delimiter
        )

    def load(self, filename: str = "replay_buffer") -> None:
        """
        Load the replay memory from files.

        Imports replay memory from numpy binary files with specified base filename.

        Parameters
        ----------
        filename : str, optional
            Base filename for loading the memory. Defaults to "replay_buffer".

        Returns
        ----------
        None
        """
        states = np.load(filename + "_states.npy")
        actions = np.load(filename + "_actions.npy")
        next_states = np.load(filename + "_next_states.npy")
        rewards = np.load(filename + "_rewards.npy")
        dones = np.load(filename + "_dones.npy")
        episodes = np.load(filename + "_episodes.npy")
        new_transitions = [
            Transition(*args)
            for args in zip(states, actions, next_states, rewards, dones, episodes)
        ]
        self.memory = deque(new_transitions, maxlen=self.capacity)

    @staticmethod
    def from_file(file_path: str) -> "ReplayMemory":
        """
        Create a ReplayMemory object from data stored in a CSV file.

        Reads replay memory data from a CSV file and creates a ReplayMemory object with it.

        Parameters
        ----------
        file_path : str
            The path to the CSV file containing the replay memory data.

        Returns
        ----------
        ReplayMemory or None
            A ReplayMemory object containing the data from the CSV file,
            or None if the file format is not supported.
        """
        print(f"[RB]: Reading data from file {file_path}...")
        if not file_path.lower().endswith(".csv"):
            print("[ERROR]: ReplayMemory.from_file only supports csv files...")
            return None
        rl_df = pd.read_csv(file_path)
        rm = ReplayMemory(capacity=len(rl_df))
        episode = 0
        for index, row in rl_df.iterrows():
            print(
                f"[RB]: loading progress {index}/{len(rl_df)} ({format(np.round(index / len(rl_df) * 100, 2), '.2f')}%)...",
                end="\r",
                flush=True,
            )
            t = Transition(
                state=row.filter(like="o_").to_numpy(),
                action=row.filter(like="a_").to_numpy(),
                next_state=row.filter(like="o2_").to_numpy(),
                reward=row["r"],
                done=row["d"],
                episode=episode,
            )
            rm.push(*t)
            if row["d"]:
                episode += 1
        print(f"[RB]: Done reading data from file {file_path}...")
        return rm

    @staticmethod
    def random_memory() -> "ReplayMemory":
        """
        Create a ReplayMemory object with random transitions.

        Returns
        ----------
        ReplayMemory
            A ReplayMemory object with random transitions.
        """
        pass

    def add_new_expert(self, sarsa_tuples: List[Tuple]) -> None:
        """
        Add new expert transitions to the memory.

        Inserts expert transitions into the memory and prints summary statistics.

        Parameters
        ----------
        sarsa_tuples : List[Tuple]
            List of SARSA tuples to add.

        Returns
        ----------
        None
        """
        self.sort_performances()
        transitions = [Transition(*args) for args in sarsa_tuples]
        transitions2 = Transition(*zip(*sarsa_tuples))
        print(transitions)
        print(transitions2)
        print(np.sum(transitions2.reward))
        print(signal.lfilter([1], [1, -0.95], transitions2.reward)[-1])

    def sort_performances(self) -> None:
        """
        Sort episodes based on their accumulated rewards.

        Accumulates and sorts episodes by total reward, then prints the sorted results.

        Returns
        ----------
        None
        """
        current_episode = self.memory[0].episode
        current_rewards = 0
        accumulated_rewards = []
        for sarsa in self.memory:
            if sarsa.episode != current_episode:
                accumulated_rewards.append(deepcopy((current_rewards, current_episode)))
                current_rewards = 0
                current_episode = sarsa.episode
            current_rewards += sarsa.reward
        accumulated_rewards.append(deepcopy((current_rewards, current_episode)))
        dtype = [("reward", float), ("episode", int)]
        accumulated_rewards = np.array(accumulated_rewards, dtype=dtype)
        sorted_rewards = np.sort(accumulated_rewards, axis=0, order="reward")
        print(accumulated_rewards)
        print(sorted_rewards.tolist())
        print("The worst expert is:", sorted_rewards[0]["episode"])


def sample_action_space(
    a_dim: int, a_max: torch.Tensor, a_min: Optional[float] = None
) -> np.ndarray:
    """
    Sample an action from the action space.

    Generates a random action within the bounds specified by `a_min` and `a_max`.

    Parameters
    ----------
    a_dim : int
        Dimension of the action space.
    a_max : torch.Tensor
        Maximum value for the action space, moved to CPU if necessary.
    a_min : Optional[float], optional
        Minimum value for the action space. Defaults to None.

    Returns
    ----------
    np.ndarray
        Sampled action.
    """
    # move from cuda to cpu
    a_max = a_max.cpu() if a_max.is_cuda else a_max

    sampled_action = (
        np.random.uniform(low=-a_max, high=a_max, size=a_dim)
        if a_min is None
        else np.random.uniform(low=a_min, high=a_max, size=a_dim)
    )
    sampled_action = np.array(sampled_action, dtype=np.float32)
    return sampled_action


def set_seed(new_seed: int) -> None:
    """
    Set the random seed for PyTorch and Numpy.

    Configures the random seed for reproducibility in PyTorch and Numpy.

    Parameters
    ----------
    new_seed : int
        The seed value to set.

    Returns
    ----------
    None
    """
    torch.manual_seed(new_seed)
    np.random.seed(new_seed)
