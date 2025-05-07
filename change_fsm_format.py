import json

original_data = {
    "states": [
        "CLOSED",
        "LISTEN",
        "SYN_SENT",
        "SYN_RCVD",
        "ESTAB",
        "FIN_WAIT-1",
        "FIN_WAIT-2",
        "CLOSING",
        "TIME-WAIT",
        "CLOSE_WAIT",
        "LAST-ACK"
    ],
    "initial_state": "CLOSED",
    "final_states": [
        "CLOSED"
    ],
    "transitions": [
        {
            "from": "CLOSED",
            "to": "LISTEN",
            "requisite": "passive OPEN",
            "actions": [
                "create TCB"
            ]
        },
        {
            "from": "CLOSED",
            "to": "SYN_SENT",
            "requisite": "active OPEN",
            "actions": [
                "create TCB",
                "send SYN"
            ]
        },
        {
            "from": "LISTEN",
            "to": "SYN_RCVD",
            "requisite": "receive SYN",
            "actions": [
                "send SYN, ACK"
            ]
        },
        {
            "from": "LISTEN",
            "to": "SYN_SENT",
            "requisite": "send",
            "actions": [
                "send SYN"
            ]
        },
        {
            "from": "LISTEN",
            "to": "CLOSED",
            "requisite": "CLOSE",
            "actions": [
                "delete TCB"
            ]
        },
        {
            "from": "SYN_SENT",
            "to": "SYN_RCVD",
            "requisite": "receive SYN",
            "actions": [
                "send SYN, ACK"
            ]
        },
        {
            "from": "SYN_SENT",
            "to": "ESTAB",
            "requisite": "receive SYN, ACK",
            "actions": [
                "send ACK"
            ]
        },
        {
            "from": "SYN_SENT",
            "to": "CLOSED",
            "requisite": "CLOSE",
            "actions": [
                "delete TCB"
            ]
        },
        {
            "from": "SYN_RCVD",
            "to": "ESTAB",
            "requisite": "receive ACK of SYN",
            "actions": []
        },
        {
            "from": "SYN_RCVD",
            "to": "FIN_WAIT-1",
            "requisite": "CLOSE",
            "actions": [
                "send FIN"
            ]
        },
        {
            "from": "SYN_RCVD",
            "to": "LISTEN",
            "requisite": "rcv RST (note1)",
            "actions": [
                ""
            ]
        },
        {
            "from": "ESTAB",
            "to": "FIN_WAIT-1",
            "requisite": "CLOSE",
            "actions": [
                "send FIN"
            ]
        },
        {
            "from": "ESTAB",
            "to": "CLOSE_WAIT",
            "requisite": "receive FIN",
            "actions": [
                "send ACK"
            ]
        },
        {
            "from": "FIN_WAIT-1",
            "to": "CLOSING",
            "requisite": "receive FIN",
            "actions": [
                "send ACK"
            ]
        },
        {
            "from": "FIN_WAIT-1",
            "to": "FIN_WAIT-2",
            "requisite": "receive ACK of FIN",
            "actions": []
        },
        {
            "from": "FIN_WAIT-2",
            "to": "TIME-WAIT",
            "requisite": "receive FIN",
            "actions": [
                "send ACK"
            ]
        },
        {
            "from": "CLOSING",
            "to": "TIME-WAIT",
            "requisite": "receive ACK of FIN",
            "actions": []
        },
        {
            "from": "TIME-WAIT",
            "to": "CLOSED",
            "requisite": "Timeout=2MSL",
            "actions": [
                "delete TCB"
            ]
        },
        {
            "from": "CLOSE_WAIT",
            "to": "LAST-ACK",
            "requisite": "CLOSE",
            "actions": [
                "send FIN"
            ]
        },
        {
            "from": "LAST-ACK",
            "to": "CLOSED",
            "requisite": "receive ACK of FIN",
            "actions": [
                "delete TCB"
            ]
        }
    ]
}

new_state_machine = {
    "states": original_data["states"],
    "initial_state": original_data["initial_state"],
    "final_states": original_data["final_states"],
    "transitions": []
}

#
def transform_transition(transition):
    from_state = transition.get("from")
    to_state = transition.get("to")
    requisite = transition.get("requisite", "").strip()
    actions = transition.get("actions", [])


    event = f"{requisite}" if requisite else ""

    if actions:
        action = "; ".join(actions)
    else:
        action = ""

    return {
        "from": from_state,
        "event": event,
        "action": action,
        "to": to_state
    }


for transition in original_data["transitions"]:
    new_transition = transform_transition(transition)
    new_state_machine["transitions"].append(new_transition)


print(json.dumps(new_state_machine, indent=2))
