import gspread
gc = gspread.service_account(filename='thesis-loi-288313-328c2d373141.json')

sheet = gc.open_by_key('11VRgwny-zURjgnyUucvGA3xvCKk0rHhPqk8YERJfqIE')
worksheet = sheet.get_worksheet(0)


def get_next_experiment(worksheet):
    sheet_values = worksheet.get_all_values()

    header = sheet_values[0]
    body = sheet_values[1:]

    experiments = [dict(zip(header, line)) for line in body]
    for i, ex in enumerate(experiments):
        ex['id'] = i + 1

    c_ex = None
    for ex in experiments:
        if ex['runned'] == 'done' or ex['runned'] == 'running':
            continue

        c_ex = ex
        break

    return c_ex, header


def set_cell(worksheet, experiment, header, key, value):
    worksheet.update_cell(experiment['id'] + 1, header.index(key) + 1, value)


experiment, header = get_next_experiment(worksheet)
if not experiment:
    print("no_experiments")
    exit()

#print(experiment)

params = experiment.copy()
out = []
naming = []
for i in params:
    valid_params = ['dataset', 'density_model', 'cc_weight', 'frames_between',
                    'epochs', 'loss_focus', 'pre', 'student_model', 'model', 'resize_patch']

    if i not in valid_params:
        continue
    if params[i]:
        out.append('--{} {}'.format(i, params[i]))

        if i == 'pre':
            naming.append('pre')
        else:
            naming.append('{}-{}'.format(i, params[i]))
params = ' '.join(out)
naming = '_'.join(naming)
from datetime import datetime
naming = '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), naming)
#print("Generated name:", naming)

# Generate name for experiment
if not experiment['name']:
    set_cell(worksheet, experiment, header, 'name', naming)
    experiment['name'] = naming

set_cell(worksheet, experiment, header, 'runned', 'running')

print("{} {}".format(experiment['name'], params))
