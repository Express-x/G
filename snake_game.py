import pygame
import random
import sys
import warnings
import time

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH = 800
HEIGHT = 600
BG_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the font
font = pygame.font.Font(None, 36)

# Set up the snake
snake = [(200, 200), (220, 200), (240, 200)]

# Set up the food
food = (400, 300)

# Set up the direction
方向 = \'right\'

# Set up the level
level = \'Easy\'

# Set up the high score
high_score = 0

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and 向轉 != \'down\':
                向轉 = \'up\'
            elif event.key == pygame.K_DOWN and 向轉 != \'up\':
                向轉 = \'down\'
            elif event.key == pygame.K_LEFT and 向轉 != \'right\':
                向轉 = \'left\'
            elif event.key == pygame.K_RIGHT and 向轉 != \'left\':
                向轉 = \'right\'
            elif event.key == pygame.K_e:
                level = \'Easy\'
            elif event.key == pygame.K_h:
                level = \'Hard\'

    # Move the snake
    head = snake[-1]
    if 向轉 == \'up\':
        new_head = (head[0], head[1] - 20)
    elif 向轉 == \'down\':
        new_head = (head[0], head[1] + 20)
    elif 向轉 == \'left\':
        new_head = (head[0] - 20, head[1])
    elif 向轉 == \'right\':
        new_head = (head[0] + 20, head[1])
    snake.append(new_head)

    # Check for collision with food
    if snake[-1] == food:
        food = (random.randint(0, WIDTH - 20) // 20 * 20, random.randint(0, WIDTH - 20) // 20 * 20)
        if level == \'Easy\':
            high_score += 1
        elif level == \'Hard\':
            high_score += 2
    else:
        snake.pop(0)

    # Check for collision with the wall or itself
    if (snake[-1][0] < 0 or snake[-1][0] >= WIDTH or
        snake[-1][1] < 0 or snake[-1][1] >= HEIGHT or
        snake[-1] in snake[:-1]):
        print(【Game Over】)
        with open(‘highscore.txt’, ‘w’) as file:
            file.write(str(high_score))
        pygame.quit()
        sys.exit(0)

    # Draw everything
    screen.fill(BG_COLOR)
    for pos in snake:
        pygame.draw.rect(screen, SNAKE_COLOR, pygame.Rect(pos[0], pos[1], 20, 20))
    pygame.draw.rect(screen, FOOD_COLOR, pygame.Rect(food[0], food[1], 20, 20))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    time.sleep(0.1)
    pygame.time.Clock().tick(10)