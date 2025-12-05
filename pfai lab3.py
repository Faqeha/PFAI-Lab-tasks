


JUG_X = 4
JUG_Y = 3


GOAL = 2


visited = set()


def apply_rules(state):
    x, y = state
    rules = []

    
    rules.append(((JUG_X, y), "Fill Jug X"))

    
    rules.append(((x, JUG_Y), "Fill Jug Y"))

    rules.append(((0, y), "Empty Jug X"))

    
    rules.append(((x, 0), "Empty Jug Y"))

   
    new_x = max(0, x - (JUG_Y - y))
    new_y = min(JUG_Y, x + y)
    rules.append(((new_x, new_y), "Pour Jug X → Jug Y"))

    
    new_x = min(JUG_X, x + y)
    new_y = max(0, y - (JUG_X - x))
    rules.append(((new_x, new_y), "Pour Jug Y → Jug X"))

    return rules


def dfs(state):
    if state in visited:
        return False

    visited.add(state)
    print(f"Current State: {state}")

    if state[0] == GOAL:
        print("\n Goal reached!")
        return True

    for new_state, action in apply_rules(state):
        print(f"→ Applying rule: {action} → New State: {new_state}")

        if dfs(new_state):
            return True

    return False


initial_state = (0, 0)

print("Water Jug Problem using DFS\n")
dfs(initial_state)
