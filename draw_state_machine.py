import json
from graphviz import Digraph

def draw_state_machine_dot(json_file,
                          output_name='state_machine_picture',
                          fmt='png',
                          dpi=300):
    """
    Reads a JSON file describing an FSM in the format:
    {
      "states": [...],
      "initial_state": "...",
      "final_states": [...],
      "transitions": [
        {
          "from": "...",
          "event": "...",
          "action": "...",
          "to": "..."
        },
        ...
      ]
    }
    and emits a Graphviz diagram (e.g. fsm.png or fsm.pdf).
    """
    # Load JSON
    with open(json_file) as f:
        data = json.load(f)

    dot = Digraph(name='ProtocolStateMachine', format=fmt)
    dot.attr(rankdir='LR', size='12,8', dpi=str(dpi))   # left-to-right layout with high DPI

    # Invisible start node pointing at the initial state
    dot.node('_start', shape='point')
    dot.edge('_start', data['initial_state'], label='')

    # Add state nodes
    finals = set(data.get('final_states', []))
    for state in data['states']:
        shape = 'doublecircle' if state in finals else 'circle'
        dot.node(state, shape=shape)

    # Add transitions
    for t in data['transitions']:
        frm = t['from']
        to = t['to']
        label_parts = []
        if t.get('event'):
            label_parts.append(t['event'])
        if t.get('action'):
            label_parts.append(t['action'])
        label = '\\n'.join(label_parts)
        dot.edge(frm, to, label=label)

    # Render to file (fsm.png or fsm.pdf)
    out_path = dot.render(filename=output_name, cleanup=True)
    print(f"FSM diagram saved to {out_path}")

if __name__ == '__main__':
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'state_machine.json'
    draw_state_machine_dot(json_path)