import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the font
font = pygame.font.Font(None, 36)

# Set up the snake and food
snake = [(200, 200), (220, 200), (240, 200)]
food = (400, 300)

# Set up the direction
direction = 'right'

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != 'down':
                direction = 'up'
            elif event.key == pygame.K_DOWN and direction != 'up':
                direction = 'down'
            elif event.key == pygame.K_LEFT and direction != 'right':
                direction = 'left'
            elif event.key == pygame.K_RIGHT and direction != 'left':
                direction = 'right'

    # Move the snake
    head = snake[-1]
    if direction == 'up':
        new_head = (head[0], head[1] - 20)
    elif direction == 'down':
        new_head = (head[0], head[1] + 20)
    elif direction == 'left':
        new_head = (head[0] - 20, head[1])
    elif direction == 'right':
        new_head = (head[0] + 20, head[1])

    snake.append(new_head)

    # Check for collisions
    if (snake[-1][0] < 0 or snake[-1][0] >= WIDTH or
        snake[-1][1] < 0 or snake[-1][1] >= HEIGHT or
        snake[-1] in snake[:-1]):
        print('Game Over')
        pygame.quit()
        sys.exit()

    # Check for food
    if snake[-1] == food:
        food = (random.randint(0, WIDTH - 20) // 20 * 20,
               random.randint(0, HEIGHT - 20) // 20 * 20)
    else:
        snake.pop(0)

    # Draw everything
    screen.fill(BACKGROUND_COLOR)
    for x, y in snake:
        pygame.draw.rect(screen, SNAKE_COLOR, (x, y, 20, 20))
    pygame.draw.rect(screen, FOOD_COLOR, (food[0], food[1], 20, 20))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.delay(100)import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 600
SPEED = 10
SCORE_FONT = pygame.font.SysFont('Arial', 24)

# Set up some variables
snake = [(200, 200), (220, 200), (240, 200)]
food = (400, 300)
score = 0
direction = 'right'

# Set up the game window
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != 'down':
                direction = 'up'
            elif event.key == pygame.K_DOWN and direction != 'up':
                direction = 'down'
            elif event.key == pygame.K_LEFT and direction != 'right':
                direction = 'left'
            elif event.key == pygame.K_RIGHT and direction != 'left':
                direction = 'right'

    # Move the snake
    head = snake[-1]
    if direction == 'up':
        new_head = (head[0], head[1] - SPEED)
    elif direction == 'down':
        new_head = (head[0], head[1] + SPEED)
    elif direction == 'left':
        new_head = (head[0] - SPEED, head[1])
    elif direction == 'right':
        new_head = (head[0] + SPEED, head[1])
    snake.append(new_head)

    # Check for collisions with the food
    if snake[-1] == food:
        score += 1
        food = (random.randint(0, WIDTH - SPEED) // SPEED * SPEED,
               random.randint(0, HEIGHT - SPEED) // SPEED * SPEED)
    else:
        snake.pop(0)

    # Check for collisions with the edge
    if (snake[-1][0] < 0 or snake[-1][0] >= WIDTH or
        snake[-1][1] < 0 or snake[-1][1] >= HEIGHT):
        print('Game Over! Final score:', score)
        pygame.quit()
        sys.exit()

    # Draw everything
    window.fill((0, 0, 0))
    for pos in snake:
        pygame.draw.rect(window, (0, 255, 0), (pos[0], pos[1], SPEED, SPEED))
    pygame.draw.rect(window, (255, 0, 0), (food[0], food[1], SPEED, SPEED))
    score_text = SCORE_FONT.render('Score: ' + str(score), True, (255, 255, 255))
    window.blit(score_text, (10, 10))
    pygame.display.update()

    # Cap the frame rate
    pygame.time.delay(1000 // 60)
import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the game window
window_width = 600
window_height = 400
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Snake Game")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)

# Snake settings
snake_block_size = 10
snake_speed = 15
snake_x = window_width / 2
snake_y = window_height / 2
snake_x_change = 0
snake_y_change = 0
snake_list = []
snake_length = 1

# Food settings
food_x = round(random.randrange(0, window_width - snake_block_size) / 10.0) * 10.0
food_y = round(random.randrange(0, window_height - snake_block_size) / 10.0) * 10.0

# Game loop
game_over = False
game_close = False

# Function to draw the snake
def draw_snake(snake_block_size, snake_list):
    for x in snake_list:
        pygame.draw.rect(window, green, [x[0], x[1], snake_block_size, snake_block_size])

# Function to draw the food
def draw_food(food_x, food_y, snake_block_size):
    pygame.draw.rect(window, red, [food_x, food_y, snake_block_size, snake_block_size])

# Game loop
while not game_over:

    while game_close == True:
        window.fill(black)
        message("You Lost! Press Q-Quit or C-Play Again", red)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
                game_close = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game_over = True
                    game_close = False
                if event.key == pygame.K_c:
                    game_loop()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                snake_x_change = -snake_block_size
                snake_y_change = 0
            elif event.key == pygame.K_RIGHT:
                snake_x_change = snake_block_size
                snake_y_change = 0
            elif event.key == pygame.K_UP:
                snake_y_change = -snake_block_size
                snake_x_change = 0
            elif event.key == pygame.K_DOWN:
                snake_y_change = snake_block_size
                snake_x_change = 0

    if snake_x >= window_width or snake_x < 0 or snake_y >= window_height or snake_y < 0:
        game_close = True

    snake_x += snake_x_change
    snake_y += snake_y_change
    window.fill(black)
    pygame.draw.rect(window, red, [food_x, food_y, snake_block_size, snake_block_size])
    snake_head = []
snake_head.append(snake_x)
snake_head.append(snake_y)
snake_list.append(snake_head)

    if len(snake_list) > snake_length:
        del snake_list[0]

    for x in snake_list[:-1]:
        if x == snake_head:
            game_close = True

    draw_snake(snake_block_size, snake_list)
    pygame.display.update()

    if snake_x == food_x and snake_y == food_y:
        food_x = round(random.randrange(0, window_width - snake_block_size) / 10.0) * 10.0
        food_y = round(random.randrange(0, window_height - snake_block_size) / 10.0) * 10.0
        snake_length += 1

    pygame.time.Clock().tick(snake_speed)

pygame.quit()
quit()

def message(msg, color):
    mesg = pygame.font.SysFont(None, 60)
    render = mesg.render(msg, True, color)
    window.blit(render, [window_width / 6, window_height / 3])

def game_loop():
    global snake_x
    global snake_y
    global snake_x_change
    global snake_y_change
    global snake_list
    global snake_length
    global food_x
    global food_y
    global game_close
    snake_x = window_width / 2
    snake_y = window_height / 2
    snake_x_change = 0
    snake_y_change = 0
    snake_list = []
    snake_length = 1
    food_x = round(random.randrange(0, window_width - snake_block_size) / 10.0) * 10.0
    food_y = round(random.randrange(0, window_height - snake_block_size) / 10.0) * 10.0
    game_close = False
    game_loop()