import asyncio
import threading
from typing import Any

import numpy as np
import pytest

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.context import Context
from tail_lib.context import Results
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityInput
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tests.helpers import returns_series
from tests.helpers import run_node
from tests.helpers import simulated_returns


class _CountingVolatility(ConstantVolatility):
    n_calls = 0

    def compute(self, returns: TimeSeries, /) -> TimeSeries:
        _CountingVolatility.n_calls += 1
        return super().compute(returns)


def _constant(returns: TimeSeries, value: float) -> TimeSeries:
    return TimeSeries(dates=returns.dates, values=np.full(returns.values.shape, value))


# --- Binding ----------------------------------------------------------------


def test_nodes_built_in_the_block_bind_to_the_context() -> None:
    outside = ConstantVolatility(0.02)
    with Context() as ctx:
        vol = EwmaVolatility(ConstantDecay(0.94))
        GaussianModel(cf=0.99, vol=vol, name="gauss")
        GaussianModel(cf=0.95, vol=outside, name="gauss_outside")

    assert Context.current() is None
    assert set(ctx.models) == {"gauss", "gauss_outside"}
    # The node built outside belongs to none, but is still read.
    nodes = {n.node for n in ctx.graph().nodes}
    assert {vol, ConstantDecay(0.94), outside} <= nodes


def test_nested_contexts_bind_to_the_innermost_one() -> None:
    with Context() as outer:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="a")
        with Context() as inner:
            GaussianModel(cf=0.99, vol=ConstantVolatility(0.02), name="b")
            assert Context.current() is inner
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.03), name="c")

    assert set(outer.models) == {"a", "c"}
    assert set(inner.models) == {"b"}


def test_threads_bind_to_their_own_context() -> None:
    contexts: dict[str, Context] = {}
    barrier = threading.Barrier(2)

    def declare(name: str) -> None:
        with Context() as ctx:
            # Both threads are inside their block at once.
            barrier.wait()
            GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name=name)
            barrier.wait()
        contexts[name] = ctx

    threads = [threading.Thread(target=declare, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert set(contexts["a"].models) == {"a"}
    assert set(contexts["b"].models) == {"b"}


def test_asyncio_tasks_bind_to_their_own_context() -> None:
    async def declare(name: str) -> Context:
        with Context() as ctx:
            await asyncio.sleep(0)
            GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name=name)
            await asyncio.sleep(0)
        return ctx

    async def main() -> tuple[Context, Context]:
        return await asyncio.gather(declare("a"), declare("b"))

    a, b = asyncio.run(main())

    assert set(a.models) == {"a"}
    assert set(b.models) == {"b"}


# --- Names ------------------------------------------------------------------


def test_a_model_built_in_a_context_needs_a_name() -> None:
    with Context(), pytest.raises(ValueError, match="needs a name"):
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))


def test_a_model_built_outside_a_context_needs_no_name() -> None:
    model = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))

    assert model.name is None


def test_two_different_nodes_cant_share_a_name() -> None:
    with Context():
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
        with pytest.raises(ValueError, match="already used"):
            GaussianModel(cf=0.95, vol=ConstantVolatility(0.01), name="gauss")
        with pytest.raises(ValueError, match="already used"):
            ConstantVolatility(0.02, name="gauss")


def test_an_equal_node_can_take_its_name_again() -> None:
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=VolatilityInput("vol"), name="a")
        GaussianModel(cf=0.95, vol=VolatilityInput("vol"), name="b")

    inputs = [n for n in ctx.graph().nodes if isinstance(n.node, VolatilityInput)]
    assert len(inputs) == 1


def test_a_node_cant_take_a_backtests_name() -> None:
    with (
        Context(backtests={"kupiec": KupiecTest}),
        pytest.raises(ValueError, match="both a node and a backtest"),
    ):
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="kupiec")


def test_a_backtest_name_has_to_be_an_identifier() -> None:
    with pytest.raises(ValueError, match="identifier"):
        Context(backtests={"kupiec-250": KupiecTest})


def test_labels_name_a_model_or_a_backtest() -> None:
    with Context(labels={"gauss": "Gaussian", "other": "?"}) as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")

    with pytest.raises(ValueError, match="name no model or backtest"):
        ctx.graph()


def test_labels_default_to_the_names() -> None:
    with Context(
        backtests={"kupiec": KupiecTest}, labels={"gauss": "Gaussian 99%"}
    ) as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
        HistoricalSimulationsModel(cf=0.99, window=Trailing(length=50), name="hs")

    assert dict(ctx.labels) == {
        "gauss": "Gaussian 99%",
        "hs": "hs",
        "kupiec": "kupiec",
    }


# --- Graph ------------------------------------------------------------------


def test_the_graph_has_paths_from_models_backtests_then_other_nodes() -> None:
    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        vol = ConstantVolatility(0.01)
        gauss = GaussianModel(cf=0.99, vol=vol, name="gauss")
        lone = EwmaVolatility(ConstantDecay(0.94), name="lone")
        EwmaVolatility(ConstantDecay(0.97))
        EwmaVolatility(ConstantDecay(0.9))

    paths = {n.node: n.paths for n in ctx.graph().nodes}

    assert paths[gauss] == ("models.gauss", "backtests.gauss.kupiec.overshootings.var")
    assert paths[vol][0] == "models.gauss.vol"
    assert paths[lone] == ("nodes.lone",)
    assert paths[ConstantDecay(0.94)] == ("nodes.lone.decay",)
    # Unnamed nodes by class, the latest built first.
    assert paths[EwmaVolatility(ConstantDecay(0.9))] == ("nodes.EwmaVolatility",)
    assert paths[EwmaVolatility(ConstantDecay(0.97))] == ("nodes.EwmaVolatility_2",)


def test_the_graph_computes_nothing() -> None:
    _CountingVolatility.n_calls = 0
    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        GaussianModel(cf=0.99, vol=_CountingVolatility(0.01), name="gauss")

    ctx.graph()

    assert _CountingVolatility.n_calls == 0


def test_backtests_reject_a_model_with_a_last_window() -> None:
    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        vol = EwmaVolatility(EwmaDecayCalibration(Last(length=100)))
        GaussianModel(cf=0.99, vol=vol, name="today")

    with pytest.raises(ValueError, match=r"`models\.today\.vol\.decay\.window`"):
        ctx.graph()


def test_two_different_nodes_cant_read_one_input() -> None:
    # Built outside the block, so its name isn't checked against the
    # context's when it is built.
    var = ValueAtRiskInput("x", cf=0.99)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=VolatilityInput("x"), name="gauss")
        KupiecTest(Overshootings(var))

    with pytest.raises(ValueError, match="read by two different nodes"):
        ctx.graph()


# --- Running ----------------------------------------------------------------


def test_run_computes_each_node_once() -> None:
    _CountingVolatility.n_calls = 0
    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        vol = _CountingVolatility(0.01)
        GaussianModel(cf=0.99, vol=vol, name="a")
        # Equal models under two names are one node.
        GaussianModel(cf=0.99, vol=vol, name="b")
        GaussianModel(cf=0.95, vol=vol, name="c")

    results = ctx.run(returns_series(n=100))

    assert _CountingVolatility.n_calls == 1
    assert results["a"] is results["b"]
    assert results.backtests["a"]["kupiec"] is results.backtests["b"]["kupiec"]


def test_results_are_read_by_node_or_by_name() -> None:
    returns = returns_series(n=100)
    with Context() as ctx:
        vol = ConstantVolatility(0.01, name="vol")
        mean = ZeroMean()
        GaussianModel(cf=0.99, vol=vol, mean=mean, name="gauss")

    results = ctx.run(returns)

    np.testing.assert_array_equal(results[vol].values, 0.01)
    assert results["vol"] is results[vol]
    assert results[mean] is results.outputs[ZeroMean()]
    with pytest.raises(KeyError, match="No node is named 'other'"):
        results["other"]
    with pytest.raises(KeyError, match="wasn't computed"):
        results[ConstantVolatility(0.5)]


def test_results_are_a_results() -> None:
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")

    assert isinstance(ctx.run(returns_series(n=50)), Results)


def test_a_context_without_models_computes_its_nodes() -> None:
    returns = simulated_returns(n=100)
    with Context() as ctx:
        vol = EwmaVolatility(ConstantDecay(0.94))

    results = ctx.run(returns)

    assert results.models == {}
    np.testing.assert_array_equal(results[vol].values, run_node(vol, returns).values)


def test_run_inputs_are_read_never_computed() -> None:
    returns = returns_series(n=100)
    vols = _constant(returns, 0.02)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=VolatilityInput("vol_ext"), name="gauss")
        ValueAtRiskInput("var_ext", cf=0.99)

    results = ctx.run(returns, {"vol_ext": vols, "var_ext": vols, "unused": vols})

    assert results[VolatilityInput("vol_ext")] is vols
    # A value-at-risk input is a model, named after its input.
    assert results.models["var_ext"] is vols


def test_missing_inputs_are_reported_before_computing() -> None:
    _CountingVolatility.n_calls = 0
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_CountingVolatility(0.01), name="gauss")
        GaussianModel(cf=0.99, vol=VolatilityInput("vol_ext"), name="ext")

    with pytest.raises(KeyError, match=r"Missing inputs \['vol_ext'\]"):
        ctx.run(returns_series(n=50))
    assert _CountingVolatility.n_calls == 0


def test_an_input_not_matching_the_returns_is_rejected() -> None:
    returns = returns_series(n=50)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=VolatilityInput("vol_ext"), name="gauss")

    with pytest.raises(ValueError, match="Input 'vol_ext' does not match"):
        ctx.run(returns, {"vol_ext": _constant(returns[:40], 0.01)})


def test_an_output_not_matching_the_returns_is_rejected() -> None:
    class _Short(ConstantVolatility):
        def compute(self, returns: TimeSeries, /) -> TimeSeries:
            return super().compute(returns[1:])

    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_Short(0.01), name="gauss")

    with pytest.raises(ValueError, match="output of _Short does not match"):
        ctx.run(returns_series(n=50))


def test_only_backtests_can_output_no_time_series() -> None:
    class _Number(ConstantVolatility):
        # Not a time series, against its own type: the run catches it.
        def compute(self, returns: TimeSeries, /) -> Any:  # noqa: ANN401, ARG002
            return 0.01

    with Context() as ctx:
        _Number(0.01)

    with pytest.raises(TypeError, match="holds no time series"):
        ctx.run(returns_series(n=50))


def test_a_run_doesnt_bind_the_backtest_nodes_it_builds() -> None:
    open_contexts: list[Context | None] = []

    def kupiec(ovs: Overshootings) -> KupiecTest:
        open_contexts.append(Context.current())
        return KupiecTest(ovs)

    with Context(backtests={"kupiec": kupiec}) as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
        # Inside the block: the backtest nodes are built outside any context.
        ctx.run(returns_series(n=50))

    assert open_contexts
    assert set(open_contexts) == {None}
