"""Original request-level temperature selection (not per isolated Score row)."""


def request_temperature(config, questions):
    if len(questions) != 1:
        raise ValueError('Independent requests require one question')
    kind = next(iter(questions.values()))['type']
    return config.get('temperature_by_type', {}).get(kind, config['temperature'])
