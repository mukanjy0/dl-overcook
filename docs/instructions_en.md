## **Final Project**

The final project consists of designing an autonomous agent capable of collaborating with another agent in the Overcooked-AI environment. The objective is to prepare and deliver as many soups as possible within a time-limited episode.

Each group will submit a single agent to play Overcooked-AI. In each scenario, the agent will be evaluated together with a partner defined for that round. The goal is to achieve the highest possible score by preparing and delivering soups.

Each scenario will have three attempts using three different seeds. The official score for the scenario will be the average of the three attempts. In some scenarios, the agent's ability to switch roles will also be evaluated.

**Score:**

```text
Score = 10000 × soups + 10 × (horizon - timestep of last soup) + (horizon - timestep of first soup) - penalty
```

If no soup is delivered, the score for that attempt will be 0.

**Penalty:**

```text
Penalty = min(100 × timeouts, 5000)
```

The number of soups is the primary factor. Time serves as a tiebreaker between agents that deliver the same number of soups. Penalties only apply to technical execution errors, such as exceeding the maximum time allowed to choose an action.

Scenarios 1, 2, and 3 will be known in advance. Scenarios 4, 5, and 6 will use new layouts and will be revealed during the competition.

Each scenario awards a maximum grade. The group keeps the highest grade it achieves.

### **Scenario 1**

- Layout: `asymmetric_advantages`
- Partner: `greedy_full_task`
- If no soup is delivered: 0
- If at least one soup is delivered: 6
- Places 15–11: 7
- Places 10–6: 8
- Places 5–1: 9
- All groups that deliver at least one soup advance.

### **Scenario 2**

- Layout: `coordination_ring`
- Partner: `greedy_full_task` with sticky actions
- If at least one soup is delivered: 9
- Places 15–11: 10
- Places 10–6: 11
- Places 5–1: 12
- All groups that deliver an average of at least two soups advance.

### **Scenario 3**

- Layout: `counter_circuit`
- Partner: `greedy_full_task` with sticky actions and random actions
- If an average of at least two soups is delivered: 11
- Places 12–9: 12
- Places 8–5: 13
- Places 4–1: 14
- Only places 1–12 qualify for the next scenario.

### **Scenario 4**

- Layout: `configs/layouts/scenario_4.layout`
    - Just change this:
    ```
    environment:
      layout_name: null
      layout_file: configs/layouts/scenario_4.layout
      horizon: 400
      old_dynamics: true
    ```
- Partner: `random_motion`
- If an average of at least one soup is delivered: 12
- Places 8–12: 14
- Places 4–8: 15
- Places 1–4: 16
- Only places 1–8 qualify for the next scenario.

### **Scenario 5**

- Layout: `<Revealed during the competition>`
- Partner: an agent from another group
- Places 5–6 and an average of at least two soups: 16
- Places 1–4 and an average of at least three soups: 17
- Only places 1–3 qualify for the final scenario.

### **Scenario 6**

- Layout: `<Revealed during the competition>`
- Partner: an agent from another group
- 3rd place and an average of at least one soup: 18
- 2nd place and an average of at least two soups: 19
- 1st place and an average of at least two soups: 20 + sublime
