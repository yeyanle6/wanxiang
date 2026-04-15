from agentforge.server.events import RunEvent


def test_run_started_includes_llm_mode_fields_when_provided() -> None:
    event = RunEvent.run_started(
        "run-1",
        plan={"workflow": "pipeline", "execution_order": []},
        llm_mode_configured="auto",
        llm_mode_effective="cli",
    )
    data = event.to_dict()["data"]
    assert data["llm_mode_configured"] == "auto"
    assert data["llm_mode_effective"] == "cli"
