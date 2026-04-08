# CrewAI: Role-Based Multi-Agent Collaboration Framework
> Open Source Framework | Date: 20260409

## Core Contribution
Orchestration framework for autonomous AI agents that collaborate through explicit role assignment and task decomposition, rather than monolithic prompts.

## Key Techniques
- **Role-based architecture**: Each agent has specialized responsibilities and tools
- **Autonomous delegation**: Agents can delegate subtasks to other agents
- **Event-driven workflows**: Crews (collaborative) and Flows (sequential/event-driven)
- **Independent architecture**: Standalone framework, not dependent on LangChain

## Architecture
- Agent: Role + Goal + Backstory + Tools
- Task: Description + Expected Output + Agent Assignment
- Crew: Group of agents working on related tasks
- Flow: Event-driven orchestration of multiple crews

## Industrial Implications
- Clean task decomposition reduces debugging complexity
- Natural mapping to team-based workflows
- Faster execution through parallel agent execution and optimized resource usage

## Interview Points
- Q: Multi-agent vs single-agent? A: Role specialization improves accuracy; delegation enables complex workflows
- Q: How to orchestrate agents? A: Role-based decomposition + event-driven flows
