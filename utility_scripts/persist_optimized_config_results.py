def persist_results(results, file):
    file_object = open(file, mode = 'w')

    for k, v in results.items():
        file_object.write(str(k) + ' >>> ' + str(v) + '\n\n')

    file_object.close()
