# Network Protocols and Their State Machine Sources

This table lists 15 common network protocols, their defining RFC documents, and sources where their state machines are described or illustrated.

| Protocol Name | RFC Document | State Machine Source |
|---------------|--------------|----------------------|
| RTSP (Real Time Streaming Protocol) | [RFC 7826](https://datatracker.ietf.org/doc/html/rfc7826) | Manually Conclude State Machine from RFC document |
| FTP (File Transfer Protocol) TODO | [RFC 959](https://datatracker.ietf.org/doc/html/rfc959) | Figure 1 of paper [Automatically Complementing Protocol Specifications From Network Traces](https://www.researchgate.net/figure/FSM-for-the-FTP-protocol-RFC-959_fig1_228795435) |
| SIP (Session Initiation Protocol) | [RFC 3261](https://datatracker.ietf.org/doc/html/rfc3261) | The state machine directly from the figure of document is not complete and only contain the state machine of one transaction process. Therefore, we use the figure 4 of article [Stateful Virtual Proxy for SIP Message Flooding Attack Detection](https://www.researchgate.net/publication/220595134_Stateful_Virtual_Proxy_for_SIP_Message_Flooding_Attack_Detection). |
| 	SMTP (Simple Mail Transfer Protocol)| [RFC 5321](https://datatracker.ietf.org/doc/html/rfc5321) | The state machine is manually extracted from section 3 of RFC 5321 |
| DCCP (Datagram Congestion Control Protocol)| [RFC 4340](https://datatracker.ietf.org/doc/html/rfc4340) | [RFCNLP](https://github.com/RFCNLP/RFCNLP) DCCP Canonical FSM, the canonical FSM is a complete FSM from not only documents.  |
| TCP (Transmission Control Protocol) | [RFC 9293](https://datatracker.ietf.org/doc/html/rfc9293) | We finds that [RFCNLP](https://github.com/RFCNLP/RFCNLP) TCP Canonical FSM is not complete compared with Figure 5 of the rfc 9293, so we use the RFC 9293 figure 5 as the gold state machine of TCP. |
| DHCPv4 (Dynamic Host Configuration Protocol for IPV4) | [RFC 2131](https://datatracker.ietf.org/doc/html/rfc2131) | The state machine is from Figure 5 of this RFC file. This may be incomplete, but I found that the state machine in [Understanding the Detailed Operations of DHCP](https://www.netmanias.com/en/post/techdocs/5999/dhcp-network-protocol/understanding-the-detailed-operations-of-dhcp) is not complete compared with Figure of the RFC file. |
| IMAP (Internet Message Access Protocol) | [RFC 9051](https://datatracker.ietf.org/doc/html/rfc9051) | Manually extract the state machine from the section 3 of RFC file. |
| POP3 (Post Office Protocol v3)| [RFC 1939](https://www.ietf.org/rfc/rfc1939.txt)| It's manually concluded from RFC file. |
| NNTP (Network News Transfer Protocol)| [RFC 3977](https://datatracker.ietf.org/doc/html/rfc3977) | manually extracted from rfc specification file. |
| MQTT  (Message Queuing Telemetry Transport) | [RFC 9431](https://datatracker.ietf.org/doc/html/rfc9431#name-reduced-protocol-interactio) |manually extracted from rfc specification file, section 2 to section 6|




The following table is the protocols that are not included in the dataset, and the reason will be shown in the table.
| Protocol Name | RFC Document | State Machine Source |
|---------------|--------------|----------------------|
| DAAP (Digital Audio Access Protocol)| It is not standardized by IETF, doesn't have a zRFC document. |  |
| HTTP/1.1 (Hypertext Transfer Protocol)| [RFC9112](https://datatracker.ietf.org/doc/html/rfc9112) | Because HTTP iss a stateless application-level protocol, it doesn't have state machine. |
| DNS (Domain Name System)| DNS is a stateless protocol. | It doesn't have a state machine. |
| SNMP (Simple Network Management Protocol) | It is a stateless protocol. | It doesn't have a state machine. |
| ICMP (Internet Control Message Protocol) | ICMP is a stateless protocol. | It doesn't have a state machine. |
