## LESSONS

- Before supporting an engine update, verify existing catalog flags against the target binary's help; upstream can remove flags such as llama.cpp's `--no-mmap` and `--mlock`.
- Design hardware controls for supported users' OS-visible topology, including multi-socket bare-metal hosts, instead of treating this one-node development VM as the product limit.
- Check each engine's own build requirements; a shared tool name does not imply a shared minimum version.
