import pygame
import heapq
import numpy as np
import random
from collections import deque
import time  # לייבוא ספריית הזמן

# אתחול המשחק
pygame.init()
WIDTH, HEIGHT, GRID_SIZE = 500, 500, 25
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# הגדרת השחקן והרודף
player = pygame.Rect(50, 50, GRID_SIZE, GRID_SIZE)
chaser = pygame.Rect(400, 400, GRID_SIZE, GRID_SIZE)
speed = GRID_SIZE

dash_speed = GRID_SIZE * 2  # דש מאפשר לשחקן לנוע מהר יותר לזמן קצר
cooldown = 0  # משתנה להגבלת תדירות הדש

# יצירת מכשולים במפה
obstacles = [(100, 100), (200, 200), (300, 300), (150, 250)]


def is_valid_move(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT and (x, y) not in obstacles


# פונקציית BFS למציאת הנתיב הקצר ביותר
def bfs_path(start, goal):
    queue = deque([(start, [])])
    visited = set()
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for dx, dy in [(-speed, 0), (speed, 0), (0, -speed), (0, speed)]:
            next_pos = (x + dx, y + dy)
            if is_valid_move(*next_pos) and next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
    return []


# פונקציית A* למציאת הנתיב היעיל ביותר
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(start, goal):
    queue = []
    heapq.heappush(queue, (0, start, []))
    visited = set()
    while queue:
        cost, (x, y), path = heapq.heappop(queue)
        if (x, y) == goal:
            return path
        for dx, dy in [(-speed, 0), (speed, 0), (0, -speed), (0, speed)]:
            next_pos = (x + dx, y + dy)
            if is_valid_move(*next_pos) and next_pos not in visited:
                visited.add(next_pos)
                new_cost = cost + 1 + heuristic(next_pos, goal)
                heapq.heappush(queue, (new_cost, next_pos, path + [next_pos]))
    return []


running = True
start_time = time.time()  # זמן התחלת המשחק

while running:
    screen.fill((30, 30, 30))
    pygame.draw.rect(screen, (0, 255, 0), player)
    pygame.draw.rect(screen, (255, 0, 0), chaser)
    for obs in obstacles:
        pygame.draw.rect(screen, (100, 100, 100), (obs[0], obs[1], GRID_SIZE, GRID_SIZE))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # תנועת השחקן
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and is_valid_move(player.x - speed, player.y): player.x -= speed
    if keys[pygame.K_RIGHT] and is_valid_move(player.x + speed, player.y): player.x += speed
    if keys[pygame.K_UP] and is_valid_move(player.x, player.y - speed): player.y -= speed
    if keys[pygame.K_DOWN] and is_valid_move(player.x, player.y + speed): player.y += speed

    # דש עם מגבלת זמן
    if keys[pygame.K_SPACE] and cooldown == 0:
        player.x += dash_speed if keys[pygame.K_RIGHT] else -dash_speed if keys[pygame.K_LEFT] else 0
        player.y += dash_speed if keys[pygame.K_DOWN] else -dash_speed if keys[pygame.K_UP] else 0
        cooldown = 10
    if cooldown > 0:
        cooldown -= 1

    # בחירת שיטת הניווט
    path = bfs_path((chaser.x, chaser.y), (player.x, player.y))  # אפשר להחליף ב-bfs_path או Q-learning
    if path:
        chaser.x, chaser.y = path[0]

    # בדיקת אם הרודף נוגע בשחקן או אם השחקן נוגע במכשול
    if chaser.colliderect(player):
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        game_over_message = f"המשחק נגמר! הזמן שחלף: {elapsed_time} שניות."
        print(game_over_message)
        running = False
    if (player.x, player.y) in obstacles:
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        game_over_message = f"המשחק נגמר! הזמן שחלף: {elapsed_time} שניות."
        print(game_over_message)
        running = False

    # הצגת הודעה על המסך כאשר המשחק נגמר
    if not running:
        font = pygame.font.Font(None, 36)
        text = font.render(game_over_message, True, (255, 0, 0))
        screen.blit(text, (WIDTH // 4, HEIGHT // 2))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
