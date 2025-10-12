## Description
The game starts with the player at location [0,0] of the frozen lake grid world with the goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

Holes in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated. Randomly generated worlds will always have a path to the goal.

The player makes moves until they reach the goal or fall in a hole.

The lake is slippery (unless disabled) so the player may move perpendicular to the intended direction sometimes (see is_slippery in Argument section).

## Action Space

The action shape is (1,) in the range {0, 3} indicating which direction to move the player.

0: Move left
1: Move down
2: Move right
3: Move up

## Observation Space

The observation is a value representing the player’s current position as current_row * ncols + current_col (where both the row and col start at 0). Therefore, the observation is returned as an integer.
For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map.

## Reward Structure

Default reward schedule:

Reach goal: +1
Reach hole: 0
Reach frozen: 0
See reward_schedule for reward customization in the Argument section.

## Our Implementation & Results