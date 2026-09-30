# What surprised me, and what still concerns me

The surprise was how much the headline number hides. 85% held-out accuracy looks like a
clear win, and it is real, but it splits into 95% when the right answer is the shorter one
and 59% when it's the longer one. The loss curve, the reward curves, lint and ship all
looked healthy. I also didn't expect the memory gap to come from *ordering* in TRL rather
than from Soup's streaming. What still concerns me: I haven't seen the real Russian data,
and the messy-ticket failure modes (templated signatures, quoted history, mixed
languages) are exactly the kind that give DPO an easy shortcut.
