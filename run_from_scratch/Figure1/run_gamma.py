import argparse
import csv
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reduced_model_codebase.reduced import final_gamma, spikevalue, trace_formula_gamma

DEFAULT_SPIKE_INDICES = (1, 79, 99, 109, 120)
CSV_FIELDS = (
    "test_name",
    "test_index",
    "average_test_error",
    "std_test_error",
    "d",
    "alpha",
    "tau",
    "kappa",
    "rho",
    "numavg",
    "lambda",
    "l",
    "n",
    "k",
    "seed",
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run finite-Gamma reduced-model simulations for Figure 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--d", type=positive_int, required=True, help="token dimension")
    parser.add_argument("--alpha", type=positive_float, required=True, help="alpha = l / d")
    parser.add_argument("--tau", type=positive_float, required=True, help="tau = n / d^2")
    parser.add_argument("--kappa", type=positive_float, required=True, help="kappa = k / d")
    parser.add_argument(
        "--numavg",
        type=positive_int,
        default=1,
        help="number of independent Gamma draws to average",
    )
    parser.add_argument("--rho", type=nonnegative_float, default=0.01, help="label-noise variance")
    parser.add_argument("--lam", type=nonnegative_float, default=1e-6, help="ridge regularizer")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/Figure1/gamma_results.csv"),
        help="CSV file for summarized results",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to the output CSV instead of overwriting it",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional NumPy random seed for reproducible simulations",
    )
    parser.add_argument(
        "--spike-indices",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_SPIKE_INDICES),
        help="1-based spike locations to evaluate",
    )
    return parser.parse_args()


def validate_args(args):
    l = int(args.alpha * args.d)
    n = int(args.tau * args.d * args.d)
    k = int(args.kappa * args.d)

    if l <= 0:
        raise ValueError("alpha*d must be at least 1")
    if n <= 0:
        raise ValueError("tau*d*d must be at least 1")
    if k <= 0:
        raise ValueError("kappa*d must be at least 1")

    invalid_indices = [idx for idx in args.spike_indices if idx > args.d]
    if invalid_indices:
        raise ValueError(
            "spike indices are 1-based and must be <= d; "
            f"got {invalid_indices} with d={args.d}"
        )


def training_covariance(d):
    # this is the trace-normalised uniform-eigenvalue distribution that we use in Figure 1
    eigenvalues = np.arange(d, 0, -1, dtype=float)
    eigenvalues *= d / eigenvalues.sum()
    return np.diag(eigenvalues)


def test_covariances(Ctr, spike_indices):
    d = Ctr.shape[0]
    tests = [("pretrain", "", Ctr)]
    for index in spike_indices:
        tests.append((f"spike_{index}", index, np.diag(spikevalue(d, 0, index - 1))))
    return tests


def run_simulation(args):
    Ctr = training_covariance(args.d)
    tests = test_covariances(Ctr, args.spike_indices)
    mu = np.zeros(args.d)
    l = int(args.alpha * args.d)

    test_errors = []
    for _ in range(args.numavg):
        Gamma = final_gamma(args.d, args.tau, args.alpha, args.kappa, args.rho, Ctr, lam=args.lam)
        test_errors.append(
            [
                trace_formula_gamma(args.d, args.rho, l, mu, Ctest, Gamma)
                for _, _, Ctest in tests
            ]
        )
    test_errors = np.asarray(test_errors, dtype=float)
    return tests, test_errors.mean(axis=0), test_errors.std(axis=0)


def result_rows(args, tests, average_test_error, std_test_error):
    l = int(args.alpha * args.d)
    n = int(args.tau * args.d * args.d)
    k = int(args.kappa * args.d)

    rows = []
    for (test_name, test_index, _), average, std in zip(
        tests, average_test_error, std_test_error
    ):
        rows.append(
            {
                "test_name": test_name,
                "test_index": test_index,
                "average_test_error": float(average),
                "std_test_error": float(std),
                "d": args.d,
                "alpha": args.alpha,
                "tau": args.tau,
                "kappa": args.kappa,
                "rho": args.rho,
                "numavg": args.numavg,
                "lambda": args.lam,
                "l": l,
                "n": n,
                "k": k,
                "seed": "" if args.seed is None else args.seed,
            }
        )
    return rows


def write_csv(rows, output, append):
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not output.exists() or output.stat().st_size == 0

    with output.open(mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.seed is not None:
        np.random.seed(args.seed)

    tests, average_test_error, std_test_error = run_simulation(args)
    rows = result_rows(args, tests, average_test_error, std_test_error)
    write_csv(rows, args.output, args.append)


if __name__ == "__main__":
    main()
