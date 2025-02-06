import numpy as np
import random
from typing import List, Tuple, Dict, Union
from main import (
    FaceTurningOctahedron,
    FaceNotation,
    FaceToIndex,
    create_scrambled_fto,
    update_plot,
)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import collections
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.exceptions import NotFittedError
import time

PENALTY_PER_MOVE = 0.01
FUTURE_REWARD_WEIGHT = 0.5  # Weight for the future reward estimation


class QLearningAgent:
    def __init__(
        self,
        actions: List[int],
        alpha: float = 0.1,
        gamma: float = 0.95,
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
    ):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = None
        self.episode = 0

    def calculate_epsilon(self) -> float:
        return max(self.min_epsilon, self.initial_epsilon - self.episode * (self.initial_epsilon - self.min_epsilon) / self.epsilon_decay_steps)

    def get_state(self, fto: FaceTurningOctahedron) -> Tuple:
        state = fto.get_state_array().flatten()
        return self.to_canonical_form(state)

    def to_canonical_form(self, state: np.ndarray) -> Tuple:
        canonical_state = state  # Apply the necessary rotations to get the canonical form
        return tuple(canonical_state)

    def choose_action(self, state: Tuple) -> int:
        epsilon = self.calculate_epsilon()
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.actions)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return np.argmax(self.q_table[state])

    def learn(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool,
    ) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def train_episode(
        self, episode: int, max_moves_per_trial: int, penalty_per_move: float
    ) -> Tuple[List[FaceTurningOctahedron], List[float], List[str]]:
        self.episode = episode
        fto, scramble = create_scrambled_fto()
        move_count = 0
        rewards = []
        states = [copy.deepcopy(fto)]
        moves = []

        prev_value = self.calculate_value(fto, move_count, penalty_per_move)
        while not self.is_solved(fto) and move_count < max_moves_per_trial:
            state = self.get_state(fto)
            action = self.choose_action(state)
            face_index, counter_clockwise = self.parse_action(action)
            fto.rotate_face(face_index, counter_clockwise)
            next_state = self.get_state(fto)
            value = self.calculate_value(fto, move_count, penalty_per_move)
            reward = value - prev_value - penalty_per_move
            prev_value = value
            done = self.is_solved(fto)
            self.learn(state, action, reward, next_state, done)
            rewards.append(reward)
            move_count += 1
            states.append(copy.deepcopy(fto))
            move_notation = [
                k[0] for k, v in FaceToIndex.__dict__.items() if v == face_index
            ][0] + ("'" if counter_clockwise else "")
            moves.append(move_notation)

        print(
            f"Episode {episode + 1}, Total Reward: {sum(rewards)}, Moves: {move_count}, Epsilon: {self.calculate_epsilon():.4f}"
        )
        return states, rewards, moves

    def train(
        self,
        episodes: int = 10000,
        animate_indices: List[int] = None,
        max_moves_per_trial: int = 200,
        penalty_per_move: float = 0.01,
    ) -> None:
        self.epsilon_decay_steps = episodes

        self.history = []
        for episode in range(episodes):
            states, rewards, moves = self.train_episode(episode, max_moves_per_trial, penalty_per_move)
            self.history.append((states, rewards, moves, episode))
        if animate_indices is None:
            animate_indices = list(range(max(0, episodes - 10), episodes))
        self.animate_all(animate_indices)

    def animate_all(self, animate_indices: List[int]) -> None:
        self.episode_ranges = []
        all_states = []
        all_rewards = []
        all_moves = []
        all_episodes = []
        idx_offset = 0
        for idx in animate_indices:
            states, rewards, moves, real_ep = self.history[idx]
            start_idx = idx_offset
            end_idx = start_idx + len(states) - 1
            self.episode_ranges.append((start_idx, end_idx))
            idx_offset = end_idx + 1
            all_states.extend(states)
            all_rewards.extend(rewards)
            all_moves.extend(moves)
            all_episodes.extend([real_ep] * len(states))

        fig, (ax, reward_ax, move_ax) = plt.subplots(3, 1, figsize=(10, 15))
        ani = FuncAnimation(
            fig,
            self.update_animation,
            frames=len(all_states),
            fargs=(
                all_states,
                all_rewards,
                all_moves,
                ax,
                reward_ax,
                move_ax,
                all_episodes,
                animate_indices,
            ),
            interval=20,
            repeat=True,
        )
        # Save the animation as a GIF
        output_path = Path.home() / "Desktop" / "animation.gif"
        ani.save(output_path, writer='imagemagick', fps=20)
        plt.show()

    def update_animation(
        self,
        frame: int,
        states: List[FaceTurningOctahedron],
        rewards: List[float],
        moves: List[str],
        ax: plt.Axes,
        reward_ax: plt.Axes,
        move_ax: plt.Axes,
        all_episodes: List[int],
        animate_indices: List[int],
    ) -> None:
        fto = states[frame]
        update_plot(fto, ax, plt.gcf())
        current_episode = all_episodes[frame] + 1

        # Check if we are at the start of a new episode
        if frame == 0 or all_episodes[frame] != all_episodes[frame - 1]:
            self.move_count = 0
            self.move_list = []

        self.move_count += 1
        self.move_list.append(moves[frame])

        ax.set_title(f"Episode {current_episode}, Move {self.move_count}")

        reward_ax.clear()
        reward_ax.set_title("Reward Over Time")
        for ep_idx, (start_idx, end_idx) in enumerate(self.episode_ranges):
            if frame >= start_idx:
                ep_frame = min(frame, end_idx)
                x_vals = range(ep_frame - start_idx + 1)
                reward_ax.plot(
                    x_vals,
                    rewards[start_idx : ep_frame + 1],
                    label=f"Episode {(self.history[ep_idx][3]+1)}",
                )
        reward_ax.legend(loc="upper left")

        move_ax.clear()
        move_ax.set_title("Moves")
        move_text = "\n".join(self.move_list)
        move_ax.text(
            0.5,
            0.5,
            move_text,
            ha="center",
            va="center",
            wrap=True,
            transform=move_ax.transAxes,
        )

        if self.is_solved(fto):
            plt.pause(3)

    def calculate_value(
        self, fto: FaceTurningOctahedron, move_count: int, penalty_per_move: float
    ) -> float:
        if self.is_solved(fto):
            return 1e5
        completeness = sum(
            (max(face.count(color) for color in set(face)) / len(face))
            for face in fto.faces
        )
        return completeness# - (penalty_per_move * move_count)  # Increasing penalty for each move

    def parse_action(self, action: int) -> Tuple[int, bool]:
        face_index = action // 2
        counter_clockwise = action % 2 == 1
        return face_index, counter_clockwise

    def is_solved(self, fto: FaceTurningOctahedron) -> bool:
        return all(len(set(face)) == 1 for face in fto.faces)


def main() -> None:
    # Actions are number i in [0, 15]
    # i // 2 = face index, i % 2 = clockwise/counter-clockwise
    agent = QLearningAgent(actions=list(range(16)))
    agent.train(episodes=2000000, animate_indices=list(range(1999990, 2000000)), max_moves_per_trial=100)


if __name__ == "__main__":
    main()
