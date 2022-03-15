import cProfile

from .orchestrator import Orchestrator
from ...experience_acquisition.replay_buffers.uniform_replay_buffer import UniformReplayBuffer


class SingleProcessOrchestrator(Orchestrator):
    def create_replay_buffer(self) -> UniformReplayBuffer:
        return UniformReplayBuffer(batch_size=self.batch_size, buffer_size=self.replay_buffer_size)

    def run(self) -> None:
        if self.output_profile:
            file_name = f'{self.output_dir}/profiles/main.prof'
            cProfile.runctx('self.run_training_epochs()', globals(), locals(), filename=file_name)
        else:
            self.run_training_epochs()

    def run_training_epochs(self) -> None:
        for epoch_idx in range(self.num_of_epochs):
            stop = self.train_epoch(epoch_idx)

            self.run_episode(epoch_idx, self.evaluate_episode_length, True)

            if stop:
                break

    def gather_experience(self) -> bool:
        exp = None

        for _ in range(self.batch_size):
            exp = self.agent.step(explore=True)
            self.replay_buffer.append(exp)

            if exp.is_done:
                break

        return exp.is_done
