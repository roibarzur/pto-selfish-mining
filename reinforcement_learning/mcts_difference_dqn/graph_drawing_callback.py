import math
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.collections
import matplotlib.colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from reinforcement_learning.base.experience_acquisition.experience import Experience
from reinforcement_learning.base.training.callbacks.training_callback import TrainingCallback
from reinforcement_learning.base.utility.multiprocessing_util import get_process_name
from reinforcement_learning.mcts_difference_dqn.mcts_agent import MCTSAgent


class GraphDrawingCallback(TrainingCallback):
    def __init__(self, visualize_every_episode: bool, visualize_every_step: bool) -> None:
        self.agent = None
        self.visualize_every_step = visualize_every_step
        self.visualize_every_episode = visualize_every_episode
        self.episode_idx = 0

    def before_running(self, agent: MCTSAgent, **kwargs) -> None:
        super().before_running(**kwargs)
        self.agent = agent

    def before_episode(self, episode_idx: int, evaluation: bool, **kwargs) -> None:
        self.episode_idx = episode_idx

    def after_episode_step(self, step_idx: int, exp: Experience, evaluation: bool, **kwargs) -> bool:
        if self.visualize_every_step:
            self.visualize_tree(f'out/graphs/{get_process_name()}/tree_{self.episode_idx}_{step_idx}.png')

        return False

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        if self.visualize_every_episode:
            self.visualize_tree(f'out/graphs/{get_process_name()}/tree_{episode_idx}.png')

    def extract_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for state_node in self.agent.monte_carlo_tree_nodes.values():
            state = state_node.state
            graph.add_node(
                str(state),
                label=f'{state}\nCount: {state_node.visit_count:,}\nNN Value: {state_node.approximated_value:,.2f}',
                shape='diamond',
                penwidth=1 + 0.3 * math.log2(1 + state_node.visit_count),
                color_value=state_node.approximated_value,
            )

            if not state_node.expanded:
                continue

            for action in state_node.legal_actions:
                action_name = self.agent.simulator.action_space.index_to_element(action).name.lower()
                graph.add_node(
                    str((state_node.state, action)),
                    label=f'{action_name}\nCount: {state_node.action_counts[action]:,}\n'
                          f'MC Value: {state_node.mc_estimated_q_values[action]:.2f}',
                    shape='box',
                    penwidth=1 + 0.3 * math.log2(1 + state_node.action_counts[action]),
                    color_value=state_node.mc_estimated_q_values[action],
                )

                graph.add_edge(
                    str(state),
                    str((state_node.state, action)),
                    label=f'{state_node.prior_action_probabilities[action]:.2f}',
                    penwidth=2 * state_node.prior_action_probabilities[action] + 0.5,
                    weight=2,
                )

                transition_probabilities = state_node.action_state_transitions[action].probabilities
                for next_state in transition_probabilities.keys():
                    graph.add_edge(
                        str((state_node.state, action)),
                        str(next_state),
                        # taillabel=f'{float(transition_probabilities[next_state]):.2}',
                        penwidth=1.5,
                        weight=1,
                    )

        return graph

    @staticmethod
    def paint_graph(graph: nx.Graph) -> plt.cm.ScalarMappable:
        cmap = plt.cm.get_cmap('plasma')
        mapping = plt.cm.ScalarMappable(cmap=cmap)

        node_color_values = nx.get_node_attributes(graph, 'color_value')
        nodes = node_color_values.keys()
        color_values = list(node_color_values.values())

        rgba_colors = mapping.to_rgba(color_values)
        rgb_colors = rgba_colors[:, :3]
        hsv_colors = mpl.colors.rgb_to_hsv(rgb_colors)
        str_colors = [' '.join(f'{val:.3}' for val in hsv_color) for hsv_color in hsv_colors]

        node_colors = dict(zip(nodes, str_colors))
        nx.set_node_attributes(graph, node_colors, name='color')

        return mapping

    # noinspection SpellCheckingInspection
    def visualize_tree(self, name: str = None, dpi: float = 150) -> None:
        graph = self.extract_graph()
        mapping = self.paint_graph(graph)
        pydot_graph = nx.nx_pydot.to_pydot(graph)

        png_bytes = pydot_graph.create_png(
            prog=[
                'dot',
                '-Gnodesep=0.5',
                '-Granksep=1.0',
                # '-Elabeldistance=5',
                # '-Elabelangle=-15',
                '-Gconcentrate=true',
                '-Gmclimit=10.0',
            ])

        stream = BytesIO()
        stream.write(png_bytes)
        stream.seek(0)
        img = mpimg.imread(stream)

        plt.figure(figsize=(1.1 * img.shape[1] / dpi, 1.15 * img.shape[0] / dpi), dpi=dpi)
        plt.imshow(img, aspect='equal')

        plt.colorbar(mapping, orientation='horizontal', fraction=0.025, pad=0.025)

        ax = plt.gca()
        ax.set_axis_off()

        if name is not None:
            Path(name).parents[0].mkdir(parents=True, exist_ok=True)
            plt.savefig(name, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # noinspection SpellCheckingInspection
    def visualize_tree_matplotlib(self) -> None:
        plt.figure(figsize=(40, 40))

        graph = self.extract_graph()
        pos = nx.nx_pydot.graphviz_layout(graph, prog=['dot', '-Gnodesep=1', '-Grankdir=LR'],
                                          root=str(self.agent.simulator.initial_state))

        # pos = nx.multipartite_layout(graph, align='horizontal')
        # pos = nx.spring_layout(graph, pos=pos, iterations=5)

        cmap = plt.cm.get_cmap('plasma')

        node_labels = nx.get_node_attributes(graph, 'label')
        node_sizes = list(nx.get_node_attributes(graph, 'size').values())
        node_colors = list(nx.get_node_attributes(graph, 'color').values())

        state_nodes = [node for node, node_type in nx.get_node_attributes(graph, 'node_type').items()
                       if node_type == 'state']
        action_nodes = [node for node, node_type in nx.get_node_attributes(graph, 'node_type').items()
                        if node_type == 'action']

        state_node_sizes = list(nx.get_node_attributes(graph.subgraph(state_nodes), 'size').values())
        state_node_colors = list(nx.get_node_attributes(graph.subgraph(state_nodes), 'color').values())

        action_node_sizes = list(nx.get_node_attributes(graph.subgraph(action_nodes), 'size').values())
        action_node_colors = list(nx.get_node_attributes(graph.subgraph(action_nodes), 'color').values())

        nx.draw_networkx_nodes(graph.subgraph(state_nodes), pos, node_size=state_node_sizes,
                               node_color=state_node_colors, node_shape='d')

        nx.draw_networkx_nodes(graph.subgraph(action_nodes), pos, node_size=action_node_sizes,
                               node_color=action_node_colors)

        edge_widths = list(nx.get_edge_attributes(graph, 'width').values())

        edges = nx.draw_networkx_edges(graph, pos, node_size=node_sizes, arrowstyle="->", arrowsize=20,
                                       edge_color='k', width=edge_widths)

        nx.draw_networkx_labels(graph, pos, labels=node_labels)

        pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        plt.colorbar(pc)
        pc.set_array(np.array(node_colors))

        ax = plt.gca()
        ax.set_axis_off()
        plt.show()
