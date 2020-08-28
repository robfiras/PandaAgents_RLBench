import os
import sys
import getopt


def eval_opts_args(argv):
    """
    This function reads the list of options and arguments, sets the corresponding variables and checks if directories
    exist
    :param argv: list of options and arguments
    :return: tuple of variables to be set
    """

    # available options
    root_log_dir = None
    use_tensorboard = False
    save_weights = False
    custom_run_id = None
    load_model_run_id = None
    path_to_model = None

    try:
        opts, args = getopt.getopt(argv, "l:i:m:tw", ["log_dir=", "id=", "load_model=", "tensorboard", "save_weights"])
    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help_message()
        elif opt in ("-l", "--log_dir"):
            root_log_dir = arg
        elif opt in ("-t", "--tensorboard"):
            use_tensorboard = True
        elif opt in ("-w", "--save_weights"):
            save_weights = True
        elif opt in ("-i", "--id"):
            custom_run_id = arg
        elif opt in ("-m", "--load_model"):
            load_model_run_id = arg

    if not root_log_dir and (save_weights or use_tensorboard):
        print("You can not use tensorboard without providing a logging directory!")
        print_help_message()
        sys.exit(2)
    elif root_log_dir and not (save_weights and use_tensorboard):
        print("You have set a logging path but neither tensorboard nor saving weights are activated!")
        print("Please activate at least one of them or don't set logging path.")
        print_help_message()

    # check if logging dir exists
    if not os.path.exists(root_log_dir):
        print("Given logging directory %s does not exist. Please double check!" % root_log_dir)
        print_help_message()
        sys.exit(2)
    # check if model to be loaded exists
    if load_model_run_id:
        path_to_model = os.path.join(root_log_dir, load_model_run_id)
        if not os.path.exists(path_to_model):
            print("There is not model %s in the given logging directory %s! Please double check!" % (load_model_run_id, root_log_dir))
            sys.exit()
        else:
            path_to_model = os.path.join(path_to_model, "weights", "")
            exit_not_found = False
            # check if actor exists
            if not os.path.exists(os.path.join(path_to_model, "actor")):
                print("No actor network found for run_id %s in path %s" % (load_model_run_id, path_to_model))
                exit_not_found = True
            # check if critic exists
            if not os.path.exists(os.path.join(path_to_model, "critic")):
                print("No critic network found for run_id %s in path %s" % (load_model_run_id, path_to_model))
                exit_not_found = True
            if exit_not_found:
                sys.exit()
    return root_log_dir, use_tensorboard, save_weights, custom_run_id, path_to_model


def print_help_message():
    print("usage: main.py -l <logging_dir> -t -w -i my_custom_run_id -m my_model_name")
    print("usage:main.py --log_dir /path/to/dir --tensorboard --save_weights --id my_custom_run_id --load_model my_model_name")
