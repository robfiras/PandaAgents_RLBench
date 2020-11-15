import sys


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "Yes": True,
             "no": False, "n": False, "No": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def adjust_save_interval(save_interval, n_workers):
    """ Adjusts save_interval so that n_worker is a divider of it """
    remainder = save_interval % n_workers
    if save_interval > n_workers:
        new_save_interval = save_interval - remainder
        print("\nShortening the save interval from %i to %i \n" % (save_interval, new_save_interval))
    else:
        new_save_interval = n_workers
        print("\nLengthening the save interval from %i to %i \n" % (save_interval, new_save_interval))
    return new_save_interval
