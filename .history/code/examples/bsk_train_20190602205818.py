from __future__ import print_function
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.networks.large import layers
from dlgo import kerasutil

import sys
import h5py

def main(model_filename, num_games):
    model = None
    epochs = 40
    batch_size = 128
    # num_games = 4531

    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    input_channels = encoder.num_planes
    input_shape = (input_channels, go_board_rows, go_board_cols)

    generator = processor.load_go_data('train', num_games, use_generator=True)
    print('Train samples: ' + str(generator.get_num_samples()))
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    if model_filename == '':
        # X, y = processor.load_go_data(num_samples=4531)
        # X = X.astype('float32')
        # Y = to_categorical(y, nb_classes)

        model = Sequential()
        network_layers = layers(input_shape)
        for layer in network_layers:
            model.add(layer)
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        model.fit_generator(generator=generator.generate(batch_size, nb_classes),  # <1>
                            epochs=epochs,
                            max_queue_size=8,
                            steps_per_epoch=generator.get_num_samples() / batch_size,  # <2>
                            validation_data=test_generator.generate(batch_size, nb_classes),  # <3>
                            validation_steps=test_generator.get_num_samples() / batch_size,  # <4>
                            callbacks=[ModelCheckpoint('checkpoints/bsk_model_epoch_{epoch}.h5')])  # <5>

        model.evaluate_generator(generator=test_generator.generate(batch_size, nb_classes),
                                steps=test_generator.get_num_samples() / batch_size)

        # model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[ModelCheckpoint('checkpoints/bsk_bot_{epoch}.h5')])
    else:
        # yaml_string = open('agents/bsk_model.yml').read()
        # model = model_from_yaml(yaml_string)
        model = load_prediction_agent(h5py.File(model_filename, "r")).model
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        model.fit_generator(generator=generator.generate(batch_size, nb_classes),  # <1>
                            epochs=epochs,
                            max_queue_size=10,
                            steps_per_epoch=generator.get_num_samples() / batch_size,  # <2>
                            validation_data=test_generator.generate(batch_size, nb_classes),  # <3>
                            validation_steps=test_generator.get_num_samples() / batch_size,  # <4>
                            callbacks=[ModelCheckpoint('checkpoints/bsk_model_epoch_{epoch}.h5')])  # <5>

        model.evaluate_generator(generator=test_generator.generate(batch_size, nb_classes),
                                steps=test_generator.get_num_samples() / batch_size)

        # model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[ModelCheckpoint('checkpoints/bsk_bot_{epoch}.h5')])

    deep_learning_bot = DeepLearningAgent(model, encoder)

    weight_file = 'agents/bsk_weights.h5'
    with h5py.File(weight_file, 'w') as bsk_bot_out:
        deep_learning_bot.serialize(bsk_bot_out)
    # model.save_weights(weight_file, overwrite=True)
    yml_file = 'agents/bsk_model.yml'
    with open(yml_file, 'w') as yml:
        model_yaml = model.to_yaml()
        yml.write(model_yaml)

if __name__ == '__main__':
    args = sys.argv
    model_file = ''
    num_games = 4531
    if len(args) == 2:
        model_file = args[1]
    elif len(args) == 3:
        model_file = args[1]
        num_games = int(args[2])
    main(model_file, num_games)
