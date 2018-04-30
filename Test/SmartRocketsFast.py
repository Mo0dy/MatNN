import numpy as np
import pygame as pg
import os
from MatNN.GeneticAlgorithm import *


os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50, 50)


NUM_ROCKETS = 500
GEN_LENGTH = 300
ANGLE_CHANGE = 0.05
STEPSIZE = 5
EVOLVE_PER_SHOW = 1000
fast = True
initial_draw = True

mouse_pressed = False


pg.init()
font = pg.font.SysFont("arialblack", 12)
clock = pg.time.Clock()
screen = pg.display.set_mode((800, 800))
pg.display.set_caption("smart rockets")
colsurf = pg.Surface((800, 800))

pg.draw.rect(colsurf, (255, 255, 255), (5, 5, 790, 790), 10)
# pg.draw.circle(colsurf, (255, 255, 255), (400, 400), 200)

# col_draw_surf = pg.Surface((800, 800))
# col_draw_surf.blit(colsurf, (0, 0, 800, 800))
#
# col_arr = pg.surfarray.pixels2d(colsurf)


rockets = np.random.rand(NUM_ROCKETS, GEN_LENGTH) / 100
rock_angle = np.zeros(NUM_ROCKETS)
rock_pos = np.zeros((NUM_ROCKETS, 2))
goal_pos = np.random.rand(2) * 800
update_rockets = np.ones(NUM_ROCKETS).astype(np.bool)

refreshrate = 120

gen_alg = GeneticAlgorithm()


def hard_reset():
    global rockets, goal_pos
    rockets = np.random.rand(NUM_ROCKETS, GEN_LENGTH) / 100
    goal_pos = np.random.rand(2) * 800
    soft_reset()


def soft_reset():
    global iteration, rock_pos, rock_angle, update_rockets
    update_rockets = np.ones(NUM_ROCKETS).astype(np.bool)
    iteration = 0
    # rockets = np.random.random_integers(0, 1, NUM_ROCKETS * GEN_LENGTH).reshape(NUM_ROCKETS, GEN_LENGTH)
    rock_angle = np.zeros(NUM_ROCKETS)
    rock_pos = np.ones((NUM_ROCKETS, 2)) * 50
    # goal_pos = np.random.rand(2) * 800


iteration = 0
draw_iter = 0
draw_bool = True


def initial_draw_func():
    if mouse_pressed:
        m_pos = pg.mouse.get_pos()
        pg.draw.circle(colsurf, (255, 255, 255), m_pos, 15)

    screen.blit(colsurf, (0, 0, 800, 800))
    pg.display.flip()


def stop_drawing():
    global col_draw_surf, col_arr, initial_draw
    initial_draw = False

    col_draw_surf = pg.Surface((800, 800))
    col_draw_surf.blit(colsurf, (0, 0, 800, 800))
    col_arr = pg.surfarray.pixels2d(colsurf)


def target_on_mouse():
    global goal_pos
    hard_reset()
    goal_pos = np.array(pg.mouse.get_pos())


def draw():
    global iteration, rock_angle, rock_pos, goal_pos, update_rockets, draw_iter, draw_bool, refreshrate
    # still in genome
    if iteration < GEN_LENGTH:
        # add angle for all ones
        rock_angle[update_rockets] += rockets[update_rockets, iteration]

        # move rockets
        rock_pos[update_rockets, 0] += np.cos(rock_angle[update_rockets]) * STEPSIZE
        rock_pos[update_rockets, 1] += np.sin(rock_angle[update_rockets]) * STEPSIZE
        iteration += 1

        # check if collision with wall:
        # values of the pxarray at the positions of the rockets
        rock_pos_int = rock_pos.astype(np.int)

        col_vals = col_arr[rock_pos_int[:, 0], rock_pos_int[:, 1]].astype(np.bool)

        update_rockets[col_vals] = False

        if not fast or draw_bool:
            screen.fill((50, 50, 50))
            screen.blit(col_draw_surf, (0, 0, 800, 800))

            for i in range(NUM_ROCKETS):
                p = rock_pos[i]
                color = (255, 255, 0)
                if col_vals[i]:
                    color = (255, 0, 0)
                pg.draw.circle(screen, color, p.astype(int), 5)
            pg.draw.circle(screen, (255, 0, 0), goal_pos.astype(int), 10)

            text = font.render("m_amount %0.4f" % gen_alg.m_amount, True, (255, 255, 255))
            screen.blit(text, (670, 10))
            text = font.render("m_change %0.4f" % gen_alg.m_changse, True, (255, 255, 255))
            screen.blit(text, (670, 22))
            text = font.render("r_amount  %0.4f" % gen_alg.r_amount, True, (255, 255, 255))
            screen.blit(text, (670, 34))
            pg.display.flip()

    else:
        draw_iter += 1
        if draw_bool:
            draw_bool = False
            refreshrate = 0
        elif draw_iter == EVOLVE_PER_SHOW:
            draw_bool = True
            refreshrate = 120
            draw_iter = 0
        # calculate distance to the goal

        distances_squared = np.sum((rock_pos[:] - goal_pos) ** 2, axis=1)
        max_dist = np.max(distances_squared)
        inverted_distances = max_dist - distances_squared

        # print(sum(inverted_distances) / inverted_distances.shape[0])
        gen_alg.recombine(rockets, inverted_distances)
        soft_reset()


loop = True
while loop:
    clock.tick(refreshrate)
    for e in pg.event.get():
        if e.type == pg.QUIT:
            loop = False
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_r:
                hard_reset()
            elif e.key == pg.K_f:
                fast = not fast
            elif e.key == pg.K_RETURN:
                stop_drawing()
            elif e.key == pg.K_UP:
                gen_alg.m_amount += 0.01
                print("m_amount = ", gen_alg.m_amount)
            elif e.key == pg.K_DOWN:
                gen_alg.m_amount -= 0.01
                print("m_amount = ", gen_alg.m_amount)
            elif e.key == pg.K_LEFT:
                gen_alg.m_changse -= 0.01
                print("m = ", gen_alg.m_changse)
            elif e.key == pg.K_RIGHT:
                gen_alg.m_changse += 0.01
                print("m_changse = ", gen_alg.m_changse)
            elif e.key == pg.K_t:
                # t for target
                target_on_mouse()

        elif e.type == pg.MOUSEBUTTONDOWN:
            mouse_pressed = True
        elif e.type == pg.MOUSEBUTTONUP:
            mouse_pressed = False
    if initial_draw:
        initial_draw_func()
    else:
        draw()

pg.quit()



