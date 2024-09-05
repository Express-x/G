import pygame
import random
import sys


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
    
    
    # Check for collision with food
    if snake[-1] == food:
        food = (random.randint(0, WIDTH - 20) // 20 * 20, random.randint(0, HEIGHT - 20) // 20 * 20)
    else:
        snake.pop(0)
    
    
    # Check for collision with the wall or itself
    if (snake[-1][0] < 0 or snake[-1][0] >= WIDTH or
        snake[-1][1] < 0 or snake[-1][1] >= HEIGHT or
        snake[-1] in snake[:-1]):
        print('Game Over')
        pygame.quit()
        sys.exit()
    
    
    # Draw everything
    screen.fill(BG_COLOR)
    for pos in snake:
        pygame.draw.rect(screen, SNAKE_COLOR, pygame.Rect(pos[0], pos[1], 20, 20))
    pygame.draw.rect(screen, FOOD_COLOR, pygame.Rect(food[0], food[1], 20, 20))
    
    
    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.delay(100)