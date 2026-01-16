from concurrent.futures import ThreadPoolExecutor

import numpy as np
import quickmp
import stumpy
import pytest


@pytest.fixture(autouse=True)
def setup_quickmp():
    """Setup and teardown for each test."""
    quickmp.initialize()
    yield
    quickmp.finalize()


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_sliding_dot_product(n, m):
    T = np.random.rand(n)
    Q = np.random.rand(m)

    QT = quickmp.sliding_dot_product(T, Q)

    # Compare to naive calculation
    for i in range(n - m + 1):
        assert np.isclose(T[i:i+m] @ Q, QT[i])

    # Compare to np.convolve
    assert np.allclose(np.convolve(T, Q[::-1], mode="valid"), QT)


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_compute_mean_std(n, m):
    T = np.random.rand(n)

    mu, sigma = quickmp.compute_mean_std(T, m)

    for i in range(n - m + 1):
        assert np.isclose(np.mean(T[i:i+m]), mu[i])
        assert np.isclose(np.std(T[i:i+m]), sigma[i])


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_selfjoin(n, m):
    T = np.random.rand(n)

    mp = quickmp.selfjoin(T, m)
    mp2 = stumpy.stump(T, m)[:, 0].astype(np.float64)

    assert np.allclose(mp, mp2)


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_abjoin(n, m):
    T1 = np.random.rand(n)
    T2 = np.random.rand(n)

    mp = quickmp.abjoin(T1, T2, m)

    mp2 = stumpy.stump(T_A=T1, T_B=T2, m=m, ignore_trivial=False)[:, 0].astype(np.float64)
    assert np.allclose(mp, mp2)


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_selfjoin_unnormalized(n, m):
    T = np.random.rand(n)

    mp = quickmp.selfjoin(T, m, normalize=False)
    mp2 = stumpy.stump(T, m, normalize=False)[:, 0].astype(np.float64)

    assert np.allclose(mp, mp2)


@pytest.mark.parametrize("n,m", [(100, 10), (500, 20), (1000, 100)])
def test_abjoin_unnormalized(n, m):
    T1 = np.random.rand(n)
    T2 = np.random.rand(n)

    mp = quickmp.abjoin(T1, T2, m, normalize=False)

    mp2 = stumpy.stump(T_A=T1, T_B=T2, m=m, ignore_trivial=False, normalize=False)[:, 0].astype(np.float64)
    assert np.allclose(mp, mp2)


def test_init_finalize():
    """Test explicit init/finalize."""
    # Already initialized by fixture, finalize first
    quickmp.finalize()

    # Should be able to init again
    quickmp.initialize(device=0)

    # Double init should raise
    with pytest.raises(RuntimeError):
        quickmp.initialize()

    quickmp.finalize()

    # Double finalize should raise
    with pytest.raises(RuntimeError):
        quickmp.finalize()

    # Re-init for fixture cleanup
    quickmp.initialize()


def test_multithread_selfjoin():
    num_threads = 4
    n, m = 500, 20

    test_data = []
    for i in range(num_threads):
        np.random.seed(i)
        T = np.random.rand(n)
        expected = stumpy.stump(T, m)[:, 0].astype(np.float64)
        test_data.append((i, T, expected))

    def worker(args):
        thread_id, T, expected = args
        mp = quickmp.selfjoin(T, m, stream=thread_id)
        assert np.allclose(mp, expected), \
            f"Stream {thread_id}: max diff = {np.max(np.abs(mp - expected))}"
        return thread_id

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(worker, test_data))

    assert len(results) == num_threads
