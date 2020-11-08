from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from snake import *
import numpy as np
from nn_viz import NeuralNetworkViz
from settings import settings
from genetic_algorithm.population import Population
from decimal import Decimal


SQUARE_SIZE = (50, 50)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings, show=True, fps=200):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(240, 240, 240))
        self.setPalette(palette)
        self.settings = settings
        self._SBX_eta = self.settings['SBX_eta']
        self._mutation_bins = np.cumsum([self.settings['probability_gaussian'],
                                        self.settings['probability_random_uniform']
        ])
        self._crossover_bins = np.cumsum([self.settings['probability_SBX'],
                                         self.settings['probability_SPBX']
        ])
        self._SPBX_type = self.settings['SPBX_type'].lower()
        self._mutation_rate = self.settings['mutation_rate']

        # Determine size of next gen based off selection type
        self._next_gen_size = None
        if self.settings['selection_type'].lower() == 'plus':
            self._next_gen_size = self.settings['num_parents'] + self.settings['num_offspring']
        elif self.settings['selection_type'].lower() == 'comma':
            self._next_gen_size = self.settings['num_offspring']
        else:
            raise Exception('Selection type "{}" is invalid'.format(self.settings['selection_type']))

        
        self.board_size = settings['board_size']
        self.border = (0, 10, 0, 10)  # Left, Top, Right, Bottom
        self.snake_widget_width = SQUARE_SIZE[0] * self.board_size[0]
        self.snake_widget_height = SQUARE_SIZE[1] * self.board_size[1]

        # Allows padding of the other elements even if we need to restrict the size of the play area
        self._snake_widget_width = max(self.snake_widget_width, 620)
        self._snake_widget_height = max(self.snake_widget_height, 600)

        self.top = 150
        self.left = 150
        self.width = self._snake_widget_width + 700 + self.border[0] + self.border[2]
        self.height = self._snake_widget_height + self.border[1] + self.border[3] + 200
        
        individuals: List[Individual] = []

        for i in range(0, 640, 20):
            # individual = Snake(self.board_size, hidden_layer_architecture=self.settings['hidden_network_architecture'],
            #                   hidden_activation=self.settings['hidden_layer_activation'],
            #                   output_activation=self.settings['output_layer_activation'],
            #                   lifespan=self.settings['lifespan'],
            #                   apple_and_self_vision=self.settings['apple_and_self_vision'])
            population_folder = "population/best_snakes"
            individual_name = f'best_snake_gen{i}'
            # load network parameters from (population_folder, individual_name)/setting.json
            individual = load_snake(population_folder, individual_name, None)
            individuals.append(individual)

        self.best_fitness = 0
        self.best_score = 0

        self._current_individual = 0

        self.population = Population(individuals)

        # initial snake
        self.snake = self.population.individuals[self._current_individual]
        # initial generation
        self.current_generation = 0

        self.init_window()

        # every 1000./fps msec call the update function
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000./fps)

        if show:
            self.show()

    def init_window(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('贪吃蛇AI')
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Create the Neural Network window
        self.nn_viz_window = NeuralNetworkViz(self.centralWidget, self.snake)
        self.nn_viz_window.setGeometry(QtCore.QRect(0, 0, 600, self._snake_widget_height + self.border[1] + self.border[3] + 200))
        self.nn_viz_window.setObjectName('nn_viz_window')

        # Create SnakeWidget window
        self.snake_widget_window = SnakeWidget(self.centralWidget, self.board_size, self.snake)
        self.snake_widget_window.setGeometry(QtCore.QRect(600 + self.border[0], self.border[1], self.snake_widget_width, self.snake_widget_height))
        self.snake_widget_window.setObjectName('snake_widget_window')

        # Genetic Algorithm Stats window
        self.ga_window = GeneticAlgoWidget(self.centralWidget, self.settings)
        self.ga_window.setGeometry(QtCore.QRect(600, self.border[1] + self.border[3] + self.snake_widget_height, self._snake_widget_width + self.border[0] + self.border[2] + 100, 200-10))
        self.ga_window.setObjectName('ga_window')


    def update(self) -> None:
        self.snake_widget_window.update()
        self.nn_viz_window.update()
        # Current individual is alive
        if self.snake.is_alive:
            self.snake.move()
            if self.snake.score > self.best_score:
                self.best_score = self.snake.score
                self.ga_window.best_score_label.setText(str(self.snake.score))
        # Current individual is dead         
        else:
            # Calculate fitness of current individual
            self.snake.calculate_fitness()
            fitness = self.snake.fitness

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.ga_window.best_fitness_label.setText('{:.2E}'.format(Decimal(fitness)))

            self._current_individual += 1
            current_pop = self.settings['num_parents'] if self.current_generation == 0 else self._next_gen_size
            self.ga_window.current_individual_label.setText(
                '{}/{}'.format(self._current_individual + 1, current_pop))

            self.snake = self.population.individuals[self._current_individual]
            self.snake_widget_window.snake = self.snake
            self.nn_viz_window.snake = self.snake

    def next_generation(self):
        self._increment_generation()
        self._current_individual = 0
    
    def _increment_generation(self):
        self.current_generation += 1
        self.ga_window.current_generation_label.setText("<font color='red'>" + str(self.loaded[self.current_generation]) + "</font>")

class GeneticAlgoWidget(QtWidgets.QWidget):
    def __init__(self, parent, settings):
        super().__init__(parent)

        font = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
        font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 5)
        TOP_LEFT = Qt.AlignLeft | Qt.AlignTop

        LABEL_COL = 0
        STATS_COL = 1
        ROW = 0

        #### Generation stuff ####
        # Generation
        self._create_label_widget_in_grid('Generation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.current_generation_label = self._create_label_widget('1', font)
        grid.addWidget(self.current_generation_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Current individual
        self._create_label_widget_in_grid('Individual: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.current_individual_label = self._create_label_widget('1/{}'.format(settings['num_parents']), font)
        grid.addWidget(self.current_individual_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Best score
        self._create_label_widget_in_grid('Best Score: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.best_score_label = self._create_label_widget('0', font)
        grid.addWidget(self.best_score_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Best fitness
        self._create_label_widget_in_grid('Best Fitness: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.best_fitness_label = self._create_label_widget('{:.2E}'.format(Decimal('0.1')), font)
        grid.addWidget(self.best_fitness_label, ROW, STATS_COL, TOP_LEFT)

        ROW = 0
        LABEL_COL, STATS_COL = LABEL_COL + 2, STATS_COL + 2

        #### GA setting ####
        self._create_label_widget_in_grid(
            '遗传算法设定', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        ROW += 1

        # Selection type
        selection_type = ' '.join([word.lower().capitalize() for word in settings['selection_type'].split('_')])
        self._create_label_widget_in_grid('Selection Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(selection_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Crossover type
        prob_SBX = settings['probability_SBX']
        prob_SPBX = settings['probability_SPBX']
        crossover_type = '{:.0f}% SBX\n{:.0f}% SPBX'.format(prob_SBX*100, prob_SPBX*100)
        self._create_label_widget_in_grid('Crossover Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(crossover_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Mutation type
        prob_gaussian = settings['probability_gaussian']
        prob_uniform = settings['probability_random_uniform']
        mutation_type = '{:.0f}% Gaussian\t\n{:.0f}% Uniform'.format(prob_gaussian*100, prob_uniform*100)
        self._create_label_widget_in_grid('Mutation Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(mutation_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Mutation rate
        self._create_label_widget_in_grid('Mutation Rate:', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        mutation_rate_percent = '{:.0f}%'.format(settings['mutation_rate'] * 100)
        mutation_rate_type = settings['mutation_rate_type'].lower().capitalize()
        mutation_rate = mutation_rate_percent + ' + ' + mutation_rate_type
        self._create_label_widget_in_grid(mutation_rate, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Lifespan
        self._create_label_widget_in_grid('Lifespan: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        lifespan = str(settings['lifespan']) if settings['lifespan'] != np.inf else 'infinite'
        self._create_label_widget_in_grid(lifespan, font, grid, ROW, STATS_COL, TOP_LEFT)

        ROW = 0
        LABEL_COL, STATS_COL = LABEL_COL + 2, STATS_COL + 2

        #### NN setting ####
        self._create_label_widget_in_grid('神经网络设定', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        ROW += 1

        # Hidden layer activation
        hidden_layer_activation = ' '.join([word.lower().capitalize() for word in settings['hidden_layer_activation'].split('_')])
        self._create_label_widget_in_grid('Hidden Activation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(hidden_layer_activation, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Output layer activation
        output_layer_activation = ' '.join([word.lower().capitalize() for word in settings['output_layer_activation'].split('_')])
        self._create_label_widget_in_grid('Output Activation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(output_layer_activation, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Network architecture
        network_architecture = '[{}, {}, 4]'.format(settings['vision_type'] * 3 + 4 + 4,
                                                    ', '.join([str(num_neurons) for num_neurons in settings['hidden_network_architecture']]))
        self._create_label_widget_in_grid('NN Architecture: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(network_architecture, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Snake vision
        snake_vision = str(settings['vision_type']) + ' directions'
        self._create_label_widget_in_grid('Snake Vision: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(snake_vision, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Snake/Apple vision type
        self._create_label_widget_in_grid('Apple/Self Vision: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        apple_self_vision_type = settings['apple_and_self_vision'].lower()
        self._create_label_widget_in_grid(apple_self_vision_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1


        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 2)

        self.setLayout(grid)
        
        self.show()

    def _create_label_widget(self, string_label: str, font: QtGui.QFont) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel()
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0,0,0,0)
        return label

    def _create_label_widget_in_grid(self, string_label: str, font: QtGui.QFont, 
                                     grid: QtWidgets.QGridLayout, row: int, col: int, 
                                     alignment: Qt.Alignment) -> None:
        label = QtWidgets.QLabel()
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0,0,0,0)
        grid.addWidget(label, row, col, alignment)


class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, parent, board_size=(50, 50), snake=None):
        super().__init__(parent)
        self.board_size = board_size
        # self.setFixedSize(SQUARE_SIZE[0] * self.board_size[0], SQUARE_SIZE[1] * self.board_size[1])
        # self.new_game()
        if snake:
            self.snake = snake
        self.setFocus()

        self.draw_vision = True
        self.show()

    def new_game(self) -> None:
        self.snake = Snake(self.board_size)
    
    def update(self):
        if self.snake.is_alive:
            self.snake.update()
            self.repaint()
        else:
            # dead
            pass

    def draw_border(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(Qt.black))
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        painter.setPen(QtCore.Qt.black)
        painter.drawLine(0, 0, width, 0)
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)

    def draw_snake(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))
        # painter.setPen(QtGui.QPen(Qt.black))
        painter.setPen(pen)
        brush = QtGui.QBrush()
        brush.setColor(Qt.red)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(198, 5, 20)))
        # painter.setBrush(brush)

        def _draw_line_to_apple(painter: QtGui.QPainter, start_x: int, start_y: int, drawable_vision: DrawableVision) -> Tuple[int, int]:
            painter.setPen(QtGui.QPen(Qt.green))
            end_x = drawable_vision.apple_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
            end_y = drawable_vision.apple_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
            painter.drawLine(start_x, start_y, end_x, end_y)
            return end_x, end_y

        def _draw_line_to_self(painter: QtGui.QPainter, start_x: int, start_y: int, drawable_vision: DrawableVision) -> Tuple[int, int]:
            painter.setPen(QtGui.QPen(Qt.red))
            end_x = drawable_vision.self_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
            end_y = drawable_vision.self_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
            painter.drawLine(start_x, start_y, end_x, end_y)
            return end_x, end_y

        for point in self.snake.snake_array:
            painter.drawRect(point.x * SQUARE_SIZE[0],  # Upper left x-coord
                             point.y * SQUARE_SIZE[1],  # Upper left y-coord
                             SQUARE_SIZE[0],            # Width
                             SQUARE_SIZE[1])            # Height

        if self.draw_vision:
            start = self.snake.snake_array[0]

            if self.snake._drawable_vision[0]:
                for drawable_vision in self.snake._drawable_vision:
                    start_x = start.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                    start_y = start.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
                    if drawable_vision.apple_location and drawable_vision.self_location:
                        apple_dist = self._calc_distance(start.x, drawable_vision.apple_location.x, start.y, drawable_vision.apple_location.y)
                        self_dist = self._calc_distance(start.x, drawable_vision.self_location.x, start.y, drawable_vision.self_location.y)
                        if apple_dist <= self_dist:
                            start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)
                            start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                        else:
                            start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                            start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)

                    elif drawable_vision.apple_location:
                        start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)

                    elif drawable_vision.self_location:
                        start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                        
                    if drawable_vision.wall_location:
                        painter.setPen(QtGui.QPen(Qt.black))
                        end_x = drawable_vision.wall_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                        end_y = drawable_vision.wall_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2 
                        painter.drawLine(start_x, start_y, end_x, end_y)

    def draw_apple(self, painter: QtGui.QPainter) -> None:
        apple_location = self.snake.apple_location
        if apple_location:
            painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            painter.setPen(QtGui.QPen(Qt.black))
            painter.setBrush(QtGui.QBrush(Qt.green))

            painter.drawRect(apple_location.x * SQUARE_SIZE[0],
                             apple_location.y * SQUARE_SIZE[1],
                             SQUARE_SIZE[0],
                             SQUARE_SIZE[1])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_border(painter)
        self.draw_apple(painter)
        self.draw_snake(painter)
        
        painter.end()

    def _calc_distance(self, x1, x2, y1, y2) -> float:
        diff_x = float(abs(x2-x1))
        diff_y = float(abs(y2-y1))
        dist = ((diff_x * diff_x) + (diff_y * diff_y)) ** 0.5
        return dist

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings)
    sys.exit(app.exec_())
