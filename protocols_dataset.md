# Network Protocols and Their State Machine Sources

This table lists 15 common network protocols, their defining RFC documents, and sources where their state machines are described or illustrated.

| Protocol Name | RFC Document | State Machine Source |
|---------------|--------------|----------------------|
| RTSP (Real Time Streaming Protocol) | [RFC 7826](https://datatracker.ietf.org/doc/html/rfc7826) | Manually Conclude State Machine from RFC document |
| FTP (File Transfer Protocol) | [RFC 959](https://datatracker.ietf.org/doc/html/rfc959) | Figure 1 of paper [Automatically Complementing Protocol Specifications From Network Traces](https://www.researchgate.net/figure/FSM-for-the-FTP-protocol-RFC-959_fig1_228795435) |
| SIP (Session Initiation Protocol) | [RFC 3261](https://datatracker.ietf.org/doc/html/rfc3261) | Manually extract from figures of this RFC |
| 	SMTP (Simple Mail Transfer Protocol)| [RFC 5321](https://datatracker.ietf.org/doc/html/rfc5321) | The state machine is manually extracted from section 3 of RFC 5321 |
| DAAP (Digital Audio Access Protocol)| It is not standardized by IETF, doesn't have a zRFC document. |  |
| DCCP (Datagram Congestion Control Protocol)| [RFC 4340](https://datatracker.ietf.org/doc/html/rfc4340) | [RFCNLP](https://github.com/RFCNLP/RFCNLP) DCCP Canonical FSM, but the canonical FSM is a complete FSM from not only documents.  |
| TCP (Transmission Control Protocol) | [RFC 9293](https://datatracker.ietf.org/doc/html/rfc9293) | [RFCNLP](https://github.com/RFCNLP/RFCNLP) TCP Canonical FSM, but the canonical FSM is a complete FSM from not only documents. |
| HTTP/1.1 (Hypertext Transfer Protocol)| |  |
| DHCP (Dynamic Host Configuration Protocol) | |  |
| IMAP (Internet Message Access Protocol) |  | |
| POP3 (Post Office Protocol v3)| |  |
| DNS (Domain Name System)|  |  |
| NNTP (Network News Transfer Protocol)|  |  |
| SNMP (Simple Network Management Protocol) | | |
| ICMP (Internet Control Message Protocol) |  | |
