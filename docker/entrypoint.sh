#!/usr/bin/env bash

# Start ssh daemon on the port passed as argument, then start an interactive bash shell.
# When used to attach to a newly created container, the container will terminate when you try to
# detach (Ctrl-P, Ctrl-Q).

# ssh port setup:
echo $SSH_PORT
if [[ -n "$SSH_PORT" ]]; then
    sed -i "s/#Port 22/Port $SSH_PORT/" /etc/ssh/sshd_config
fi

# Start the ssh server:
/etc/init.d/ssh start

# This warning normally reaches the user only when triggered from `boot-*-attached`:
echo "Warning: Ctrl-P, Ctrl-Q will kill the container, not just detach from it!"

# Open a bash shell; 
#bash -l
