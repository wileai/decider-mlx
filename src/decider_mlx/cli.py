"""Command-line entry point. Example payloads are deliberately synthetic."""
import argparse
import json
from pathlib import Path
import sys


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default errors can echo arbitrary argument values.
        super().error('Invalid command-line arguments; use --help for usage')


def example_request():
    return {
        'state': {'message': 'Please explain how to reset the demo device.'},
        'questions': {'route': {
            'type': 'choice',
            'instructions': 'Which team should handle this message?',
            'criteria': {'support': 'Product help', 'sales': 'Purchasing questions'},
        }},
    }


def main(argv=None):
    parser = PrivateArgumentParser(prog='decider-mlx')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('example', help='Print a synthetic request JSON; no model needed')
    fetch = commands.add_parser('fetch-source', help='Download two hash-verified public source modules')
    fetch.add_argument('destination', type=Path)
    decide = commands.add_parser('decide', help='Run one local FP16 decision')
    decide.add_argument('--checkpoint', required=True, type=Path)
    decide.add_argument('--upstream-source', required=True, type=Path)
    decide.add_argument('--max-state-tokens', type=int, default=2048)
    inputs = decide.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--example', action='store_true', help='Use a synthetic routing request')
    inputs.add_argument('--request', type=Path, help='Read {state, questions} from a JSON file')
    inputs.add_argument('--stdin', action='store_true', help='Read {state, questions} from stdin')
    args = parser.parse_args(argv)
    failure = 'Unable to prepare the example'
    try:
        if args.command == 'example':
            result = example_request()
        elif args.command == 'fetch-source':
            failure = 'Unable to fetch verified source; check destination and network access'
            from .upstream import fetch_source, UPSTREAM_COMMIT
            fetch_source(args.destination)
            result = {'upstream_commit': UPSTREAM_COMMIT, 'verified': True}
        else:
            failure = 'Unable to read request; supply valid JSON with exactly state and questions'
            if args.example:
                request = example_request()
            elif args.stdin:
                request = json.load(sys.stdin)
            else:
                request = json.loads(args.request.read_text())
            if not isinstance(request, dict) or set(request) != {'state', 'questions'}:
                raise ValueError('Request must contain exactly state and questions')
            failure = 'Unable to initialize runtime; check installation and local model/source files'
            from . import Decider
            model = Decider(args.checkpoint, args.upstream_source,
                            max_state_tokens=args.max_state_tokens)
            failure = 'Unable to decide; check question schema and configured input limits'
            result = model.decide(request['state'], request['questions'])
        failure = 'Unable to write the response'
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        # This CLI is a privacy boundary for diagnostics, including unexpected
        # dependency errors. Never echo exception values or local tracebacks.
        parser.exit(1, f'{parser.prog}: {failure}\n')
    except KeyboardInterrupt:
        parser.exit(130, f'{parser.prog}: Interrupted\n')
    return 0
