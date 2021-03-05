from mulval import MulvalAttackGraph, MulvalVertexType


class State:
    """
    Container to represent a state in an attack graph.

    :param int id_: An auto-generated id, starting at 0.
    :param set ids_propositions: A set of ids corresponding to the propositions
    this state has.
    """
    def __init__(self, id_: int, ids_propositions: set):
        self.id_ = id_
        self.ids_propositions = ids_propositions

        # The too following dictionnaries corresponds respectively to the
        # neighbours with incoming and outgoing transitions.
        # Each key is an integer corresponding to a state id and associated
        # with a set of integers corresponding to the ids of the transitions
        # linking the two states.
        self.in_ = {}
        self.out = {}


class Transition:
    """
    Container to represent a transition in an attack graph.

    :param int id_: The id of the transition. It corresponds to the id of the
    associated AND node in a MulVAL attack graph.
    :param int state_id_from: The id of the starting state of the transition.
    :param int state_id_to: The id of the ending state of the transition.
    """
    def __init__(self, id_: int, state_id_from: int, state_id_to: int):
        self.id_ = id_
        self.state_id_from = state_id_from
        self.state_id_to = state_id_to


class Proposition:
    """
    Container to represent a proposition. A proposition is a fact that can be
    either true or false. It corresponds to a LEAF or an OR node in a MulVAL
    attack graph.

    :param int id_: The id of the transition. It corresponds to the id of the
    associated node in a MulVAL attack graph.
    :param str fact: A string describing the proposition.
    """
    def __init__(self, id_: int, fact: str):
        self.id_ = id_
        self.fact = fact


class AttackGraph:
    """
    A class to represents attack graphs.
    """
    def __init__(self):
        self.states = []
        self.transitions = []
        self.propositions = {}

        self.N = 0

    def import_mulval_attack_graph(self, mag: MulvalAttackGraph):
        """
        Convert a MulVAL attack graph into a state attack graph. In a MulVAL
        attack graph, nodes can be either proposition or transition. In a
        state attack graph, nodes are sets of propositions and edges are
        transitions.

        :param MulvalAttackGraph mag: The MulVAL attack graph to convert.
        """
        # The ids of the propositions that are true at
        # the beginning
        ids_propositions = []
        # The ids of all the possible transitions
        ids_transitions = []

        # Loop through all the nodes and fill the corresponding lists,
        # dictionnaries and sets.
        for i in range(mag.N):
            vertex = mag.vertices[mag.ids[i]]
            if vertex.type_ == MulvalVertexType.LEAF:
                # The vertex is a LEAF which corresponds to a proposition that
                # is true at the beginning
                ids_propositions.append(vertex.id_)
                self.propositions[vertex.id_] = Proposition(
                    vertex.id_, vertex.fact)
            elif vertex.type_ == MulvalVertexType.OR:
                # The vertex is an OR which corresponds to a proposition that
                # is false at the beginning
                self.propositions[vertex.id_] = Proposition(
                    vertex.id_, vertex.fact)
            else:
                # The vertex is an AND which corresponds to a transition
                ids_transitions.append(vertex.id_)

        # Fill the states and transitions
        initial_state = State(0, set(ids_propositions))
        self.states.append(initial_state)

        self.find_states_and_transitions(mag, initial_state, ids_transitions)

        # Update the number of states
        self.N = len(self.states)

        # Add the neighbours of the states
        self.add_neighbours_to_states(mag)

    def find_states_and_transitions(self, mag: MulvalAttackGraph, state: State,
                                    ids_transitions: list):
        """
        Go through the MulVAL attack graph and fill the states and
        propositions.
        This is a recursive function. At each call, it looks for the possible
        transitions depending on the set of true propositions. If a transition
        can be made, it is added to the list of transitions and the reached
        state is added to the list of states.

        :param MulvalAttackGraph mag: The MulVAL attack graph to convert.
        :param State state: The current state.
        :param list ids_transitions: The ids of the transitions that haven't
        been used.
        """
        # Loop through all the transitions to see if one is possible
        for id_transition in ids_transitions:
            # Check if this transition is possible
            vertex_and = mag.vertices[id_transition]
            possible = True
            for id_required in vertex_and.in_:
                possible = possible and id_required in state.ids_propositions

            if not possible:
                continue

            # If this transition is possible, add the reached state to the list
            # of states and the transition to the list of transitions
            granted_proposition_id = vertex_and.out[0]
            ids_propositions = set(
                [*state.ids_propositions, granted_proposition_id])

            # Add the state only if it doesn't already exist
            id_existing = -1
            i_state = 0
            while i_state < len(self.states) and id_existing == -1:
                if self.states[i_state].ids_propositions == ids_propositions:
                    id_existing = i_state
                else:
                    i_state += 1

            if id_existing == -1:
                new_state = State(len(self.states), ids_propositions)
                self.states.append(new_state)
            else:
                new_state = self.states[id_existing]

            # Check if the transition already exists
            id_existing = -1
            i_transition = 0
            while i_transition < len(self.transitions) and id_existing == -1:
                transition = self.transitions[i_transition]
                already_exists = vertex_and.id_ == transition.id_
                already_exists &= transition.state_id_from == state.id_
                already_exists &= transition.state_id_to == new_state.id_
                if already_exists:
                    id_existing = i_transition
                else:
                    i_transition += 1

            # If the transition already exists, we do not add it
            # Moreover, if the state and the new state are the same, there is
            # no point for the attacker in using the exploit
            if id_existing == -1 and state.id_ != new_state.id_:
                transition = Transition(vertex_and.id_, state.id_,
                                        new_state.id_)
                self.transitions.append(transition)

            # Recursively call this function with the new reached state and
            # with the used transition removed
            new_ids_transitions = ids_transitions.copy()
            new_ids_transitions.remove(vertex_and.id_)
            self.find_states_and_transitions(mag, new_state,
                                             new_ids_transitions)

    def add_neighbours_to_states(self, mag: MulvalAttackGraph):
        """
        Fill the dictionnaries in_ and out once the MulVAL attack graph has
        been converted.

        :param MulvalAttackGraph mag: The MulVAL attack graph to convert.
        """
        for t in self.transitions:
            if t.state_id_to in self.states[t.state_id_from].out:
                self.states[t.state_id_from].out[t.state_id_to].add(t.id_)
            else:
                self.states[t.state_id_from].out[t.state_id_to] = {t.id_}

            if t.state_id_from in self.states[t.state_id_to].in_:
                self.states[t.state_id_to].in_[t.state_id_from].add(t.id_)
            else:
                self.states[t.state_id_to].in_[t.state_id_from] = {t.id_}
