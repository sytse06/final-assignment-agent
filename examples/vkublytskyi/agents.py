from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rich.console import Console
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    OpenAIServerModel,
    AgentLogger,
    LogLevel,
    Panel,
    Text,
)
from tools import (
    GetAttachmentTool,
    GoogleSearchTool,
    GoogleSiteSearchTool,
    ContentRetrieverTool,
    YoutubeVideoTool,
    SpeechRecognitionTool,
    ClassifierTool,
    ImageToChessBoardFENTool,
    chess_engine_locator,
)
import openai
import backoff

def create_general_ai_agent(verbosity: int = LogLevel.INFO):
    get_attachment_tool = GetAttachmentTool()
    speech_recognition_tool = SpeechRecognitionTool()
    env_tools = [
        get_attachment_tool,
    ]
    model = OpenAIServerModel(model_id="gpt-4.1")
    console = Console(record=True)
    logger = AgentLogger(level=verbosity, console=console)
    steps_buffer = []


    def capture_step_log(agent) -> None:
        steps_buffer.append(console.export_text(clear=True))


    agents = {
        agent.name: agent
        for agent in [
            ToolCallingAgent(
                name="general_assistant",
                description="Answers questions for best of knowledge and common reasoning grounded on already known information. Can understand multimedia including audio and video files and YouTube.",
                model=model,
                tools=env_tools
                + [
                    speech_recognition_tool,
                    YoutubeVideoTool(
                        client=model.client,
                        speech_recognition_tool=speech_recognition_tool,
                        frames_interval=3,
                        chunk_duration=60,
                        debug=True,
                    ),
                    ClassifierTool(
                        client=model.client,
                        model_id="gpt-4.1-mini",
                    ),
                ],
                logger=logger,
                step_callbacks=[capture_step_log],
            ),
            ToolCallingAgent(
                name="web_researcher",
                description="Answers questions that require grounding in unknown information through search on web sites and other online resources.",
                tools=env_tools
                + [
                    GoogleSearchTool(),
                    GoogleSiteSearchTool(),
                    ContentRetrieverTool(),
                ],
                model=model,
                planning_interval=3,
                max_steps=9,
                logger=logger,
                step_callbacks=[capture_step_log],
            ),
            CodeAgent(
                name="data_analyst",
                description="Data analyst with advanced skills in statistic, handling tabular data and related Python packages.",
                tools=env_tools,
                additional_authorized_imports=[
                    "numpy",
                    "pandas",
                    "tabulate",
                    "matplotlib",
                    "seaborn",
                ],
                model=model,
                logger=logger,
                step_callbacks=[capture_step_log],
            ),
            CodeAgent(
                name="chess_player",
                description="Chess grandmaster empowered by chess engine. Always thinks at least 100 steps ahead.",
                tools=env_tools
                + [
                    ImageToChessBoardFENTool(client=model.client),
                    chess_engine_locator,
                ],
                additional_authorized_imports=[
                    "chess",
                    "chess.engine",
                ],
                model=model,
                logger=logger,
                step_callbacks=[capture_step_log],
            ),
        ]
    }


    class GAIATask(TypedDict):
        task_id: Optional[str | None] = None
        question: str
        steps: list[str] = []
        agent: Optional[str | None] = None
        raw_answer: Optional[str | None] = None
        final_answer: Optional[str | None] = None


    llm = ChatOpenAI(model="gpt-4.1")
    logger = AgentLogger(level=verbosity)


    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=6)
    def llm_invoke_with_retry(messages):
        response = llm.invoke(messages)
        return response


    def read_question(state: GAIATask):
        logger.log_task(
            content=state["question"].strip(),
            subtitle=f"LangGraph with {type(llm).__name__} - {llm.model_name}",
            level=LogLevel.INFO,
            title="Final Assignment Agent for Hugging Face Agents Course",
        )
        get_attachment_tool.attachment_for(state["task_id"])

        return {
            "steps": [],
            "agent": None,
            "raw_answer": None,
            "final_answer": None,
        }


    def select_agent(state: GAIATask):
        agents_description = "\n\n".join(
            [
                f"AGENT NAME: {a.name}\nAGENT DESCRIPTION: {a.description}"
                for a in agents.values()
            ]
        )

        prompt = f"""\
    You are a general AI assistant.

    I will provide you a question and a list of agents with their descriptions.
    Your task is to select the most appropriate agent to answer the question.
    You can select one of the agents or decide that no agent is needed.

    If question has attachment only agent can answer it.

    QUESTION:
    {state["question"]}

    {agents_description}

    Now, return the name of the agent you selected or "no agent needed" if you think that no agent is needed.
    """

        response = llm_invoke_with_retry([HumanMessage(content=prompt)])
        agent_name = response.content.strip()

        if agent_name in agents:
            logger.log(
                f"Agent {agent_name} selected for solving the task.",
                level=LogLevel.DEBUG,
            )
            return {
                "agent": agent_name,
                "steps": state.get("steps", [])
                + [
                    f"Agent '{agent_name}' selected for task execution.",
                ],
            }
        elif agent_name == "no agent needed":
            logger.log(
                "No appropriate agent found in the list. No agent will be used.",
                level=LogLevel.DEBUG,
            )
            return {
                "agent": None,
                "steps": state.get("steps", [])
                + [
                    "A decision is made to solve the task directly without invoking any agent.",
                ],
            }
        else:
            logger.log(
                f"[bold red]Warning to user: Unexpected agent name '{agent_name}' selected. No agent will be used.[/bold red]",
                level=LogLevel.INFO,
            )
            return {
                "agent": None,
                "steps": state.get("steps", [])
                + [
                    f"Attempt to select non-existing agent '{agent_name}'. No agent will be used.",
                ],
            }


    def delegate_to_agent(state: GAIATask):
        agent_name = state.get("agent", None)
        if not agent_name:
            raise ValueError("Agent not selected.")
        if agent_name not in agents:
            raise ValueError(f"Agent '{agent_name}' is not available.")

        logger.log(
            Panel(Text(f"Calling agent: {agent_name}.")),
            level=LogLevel.INFO,
        )

        agent = agents[agent_name]
        agent_answer = agent.run(task=state["question"])
        steps = [f"Agent '{agent_name}' step:\n{s}" for s in steps_buffer]
        steps_buffer.clear()
        return {
            "raw_answer": agent_answer,
            "steps": state.get("steps", []) + steps,
        }


    def one_shot_answering(state: GAIATask):
        response = llm_invoke_with_retry([HumanMessage(content=state.get("question"))])
        return {
            "raw_answer": response.content,
            "steps": state.get("steps", [])
            + [
                f"One-shot answer:\n{response.content}",
            ],
        }


    def refine_answer(state: GAIATask):
        question = state.get("question")
        answer = state.get("raw_answer", None)
        if not answer:
            return {"final_answer": "No answer."}

        prompt = f"""\
    You are a general AI assistant.

    I will provide you a question and correct answer to it. Answer is correct but may be too verbose or not follow the rules below.
    Your task is to rephrase answer according to rules below.

    Answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 

    If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
    If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
    If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

    If you are asked for a comma separated list, use space after comma and before next element of the list unless other directly specified in a question.
    Check question context to define if letters case matters. Do not change case if not prescribed by other rules or question.
    If you are not asked for the list, capitalize the first letter of the answer unless it changes meaning of the answer.
    If answer is number, use digits only not words unless other directly specified in a question.
    If answer is not full sentence, do not add period at the end.

    Preserve all items if the answer is a list.

    QUESTION:
    {question}

    ANSWER:
    {answer}
    """
        response = llm_invoke_with_retry([HumanMessage(content=prompt)])
        refined_answer = response.content.strip()
        logger.log(
            Text(f"GAIA final answer: {refined_answer}", style="bold #d4b702"),
            level=LogLevel.INFO,
        )
        return {
            "final_answer": refined_answer,
            "steps": state.get("steps", [])
            + [
                "Refining the answer according to GAIA benchmark rules.",
                f"FINAL ANSWER: {response.content}",
            ],
        }


    def route_task(state: GAIATask) -> str:
        if state.get("agent") in agents:
            return "agent selected"
        else:
            return "no agent matched"


    # Create the graph
    gaia_graph = StateGraph(GAIATask)

    # Add nodes
    gaia_graph.add_node("read_question", read_question)
    gaia_graph.add_node("select_agent", select_agent)
    gaia_graph.add_node("delegate_to_agent", delegate_to_agent)
    gaia_graph.add_node("one_shot_answering", one_shot_answering)
    gaia_graph.add_node("refine_answer", refine_answer)

    # Start the edges
    gaia_graph.add_edge(START, "read_question")
    # Add edges - defining the flow
    gaia_graph.add_edge("read_question", "select_agent")

    # Add conditional branching from select_agent
    gaia_graph.add_conditional_edges(
        "select_agent",
        route_task,
        {"agent selected": "delegate_to_agent", "no agent matched": "one_shot_answering"},
    )

    # Add the final edges
    gaia_graph.add_edge("delegate_to_agent", "refine_answer")
    gaia_graph.add_edge("one_shot_answering", "refine_answer")
    gaia_graph.add_edge("refine_answer", END)

    gaia = gaia_graph.compile()
    return gaia
