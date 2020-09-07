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
    no_training = False
    path_to_model = None
    training_episodes = None
    run_headless = False
    read_buffer_id = None
    write_buffer = False
    n_worker = 0


    try:
        opts, args = getopt.getopt(argv, "l:i:m:e:r:W:twnhH", ["help","log_dir=", "id=", "load_model=",
                                                               "episodes=", "tensorboard", "save_weights",
                                                               "no_training", "headless", "read_buffer=",
                                                               "Worker=", "write_buffer"])
    except getopt.GetoptError as e:
        print("\n Error: ", e, "\n")
        print_help_message()

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help_message()
            sys.exit()
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
        elif opt in ("-e", "--episodes"):
            training_episodes = int(arg)
        elif opt in ("-H", "--headless"):
            run_headless = True
        elif opt in ("-n", "--no_training"):
            no_training = True
        elif opt in ("-r", "--read_buffer"):
            read_buffer_id = arg
        elif opt in ("-w", "--write_buffer"):
            write_buffer = True
        elif opt in ("-W", "--Worker"):
            n_worker = int(arg)

    if not root_log_dir and ((save_weights or use_tensorboard or write_buffer) or read_buffer_id is not None):
        print("You can not use tensorboard, save the weights, read/save the buffer or load a model without "
              "providing a root logging directory!")
        print_help_message()

    # check if logging dir exists
    if not os.path.exists(root_log_dir):
        print("Given logging directory %s does not exist. Please double check!" % root_log_dir)
        print_help_message()
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
    # check if there exists a directory of the buffer
    path_to_read_buffer = None
    if read_buffer_id:
        path_to_read_buffer = os.path.join(root_log_dir, read_buffer_id, "")
        if not os.path.exists(path_to_read_buffer):
            raise FileNotFoundError("The given path to the read database's directory does not exists: %s" %
                                    path_to_read_buffer)

    return {"root_dir": root_log_dir,
            "use_tensorboard": use_tensorboard,
            "save_weights": save_weights,
            "run_id": custom_run_id,
            "path_to_model": path_to_model,
            "training_episodes": training_episodes,
            "no_training": no_training,
            "headless": run_headless,
            "path_to_read_buffer": path_to_read_buffer,
            "write_buffer": write_buffer,
            "n_worker": n_worker}


def print_help_message():
    print("\nUsage: main.py -l <logging_dir> -e 100 -m my_model_name \n\n"
          "Options:\n"
          "-e, --episodes <number_episodes>    Sets the number of episodes to <number_episodes>\n"
          "-l, --log_dir <logging_dir>         Sets the logging directory to <logging_dir>\n"
          "-w, --save_weights                  If set, weights are save at <logging_dir>\n"
          "-t, --tensorboard                   If set, stores tensorboard logs at <logging_dir>\n"
          "-i, --id <id>                       Sets the run-id for logging to <id>\n"
          "-m, --load_model <model_location>   Sets the location of the model to be loaded \n"
          "-H, --headless                      If set, runs Coppelia-simulator in headless-mode\n"
          "-n, --no_training                   If set, does not train the models\n"
          "-r, --read_buffer <id_buffer>       Sets the id (former run-id) of the buffer to be read\n"
          "-w, --write_buffer                  If set, the buffer is written to a sql database in the logging_dir\n"
          "-W, --Worker                        Sets the number of workers")
    sys.exit()


