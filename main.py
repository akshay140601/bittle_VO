import yaml
#from dataloader import DataLoader
from frontend import frontend

if __name__ == '__main__':

    with open("config/initial_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)

    sequence = config['data']['sequence']

    #data_handler = DataLoader(sequence=sequence)
    #data_handler.reset_frames()

    #trajectory = frontend(data_handler)
