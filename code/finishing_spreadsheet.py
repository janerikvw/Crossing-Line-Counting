import gspread
gc = gspread.service_account(filename='thesis-loi-288313-328c2d373141.json')

sheet = gc.open_by_key('11VRgwny-zURjgnyUucvGA3xvCKk0rHhPqk8YERJfqIE')
worksheet = sheet.get_worksheet(0)

sheet_values = worksheet.get_all_values()

header = sheet_values[0]
body = sheet_values[1:]


def set_cell(worksheet, experiment, header, key, value):
    worksheet.update_cell(experiment['id'] + 1, header.index(key) + 1, value)


experiments = [dict(zip(header, line)) for line in body]
for i, ex in enumerate(experiments):
    ex['id'] = i + 1

c_ex = None
for ex in experiments:
    if ex['runned'] != 'running':
        continue

    try:
        file = open('full_on_pwc/finals/{}.txt'.format(ex['name']), mode='r')
        all_of_it = file.read()
        file.close()
        # mae, mse = all_of_it.split(',')
        # mae = float(mae.strip())
        # mse = float(mse.strip())
        # print('{}: {} {}'.format(ex['name'], mae, mse))

        # set_cell(worksheet, ex, header, 'best_mae', mae)
        # set_cell(worksheet, ex, header, 'best_mse', mse)
        set_cell(worksheet, ex, header, 'runned', 'done')
    except IOError:
        print("{} still running".format(ex['name']))
        continue
    finally:
        print('Accept')
