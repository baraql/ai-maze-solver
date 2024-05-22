import os
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import pickle
# from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

from map_walls import map_walls
from radar import radar_mask
from draw_fov import draw_fov

def convert_opencv_img_to_pygame(opencv_image):
    opencv_image = opencv_image[:, :, ::-1]
    shape = opencv_image.shape[1::-1]
    pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
    
    return pygame_image

# upper bound for movement step: 5
# lower bound for movement step: 4.8

# too fast -> forward away
# too slow -> forward towards

# movement_step = 2.483875  
resize_threshold = 60

FREE = (255, 255, 255) 
OBSTACLE = (0, 0, 255)  
UNEXPLORED = (107, 107, 107) 
INITIAL_MINIMAP_SIZE = (1000, 1000)
explore_radius = 15


wall_size_threshold = 60

shift_amt = 40



def transform_points(points, camera_pos, camera_angle, scale):

    angle_rad = np.deg2rad(camera_angle)

    rotation_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [np.sin(angle_rad), -np.cos(angle_rad)] 
    ])

    rotated_points = np.dot(points, rotation_matrix) * scale

    transformed_points = rotated_points + camera_pos

    return transformed_points


class KeyboardPlayerPyGame(Player):
    def __init__(self):

        self.rotation_step = np.float64(2.44881)
        self.movement_step = np.float64(2.483875)
        
        self.wall_scale = 10
        self.fov_radius = 25
        self.wall_circle_size = 1
        
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        
        # Minimap
        self.minimap_size = INITIAL_MINIMAP_SIZE  # Size of the minimap
        self.wall_points = np.empty((0, 2))  # Array to store wall points
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for saving data
        self.count = 0  # Counter for saving images
        self.save_dir = "data/images/"  # Directory to save images to

        # Initialize SIFT detector
        # SIFT stands for Scale-Invariant Feature Transform
        self.sift = cv2.SIFT_create()
        # Load pre-trained codebook for VLAD encoding
        # If you do not have this codebook comment the following line
        # You can explore the maze once and generate the codebook (refer line 181 onwards for more)
        self.codebook = pickle.load(open("codebook-gigantic.pkl", "rb"))
        # Initialize database for storing VLAD descriptors of FPV
        self.database = []
        self.vertical_strips = []
        self.targets = None
        img_strip = None 
        # Radar
        self.at_wall = False
        self.notice_walls = True
        self.up_pressed_while_at_wall = False
        
        # Minimap
        self.camera_position = np.array([500, 500], dtype=np.float64)  # Start at the center of the grid
        self.camera_direction = np.float64(0)  # 0 = North, 90 = East, 180 = South, 270 = West
        self.scale_factor = 1
        
        self.minimap = pygame.Surface(INITIAL_MINIMAP_SIZE)
        self.minimap.fill(UNEXPLORED)
        
        self.wall_minimap = pygame.Surface(INITIAL_MINIMAP_SIZE, pygame.SRCALPHA)
        self.wall_minimap.fill((0,0,0,0))
        
        self.character_layer = pygame.Surface(INITIAL_MINIMAP_SIZE, pygame.SRCALPHA)
        self.character_layer.fill((0, 0, 0, 0))  # Fill with transparent color
        
        self.coords = []
        self.path_layer = None
        
        self.transform = [0, 0]
        
        self.scale_threshold = 60
        
        # image_files = [f for f in os.listdir(self.save_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # for image_file in image_files:
        #     # Construct the full path to the image
        #     image_path = os.path.join(self.save_dir, image_file)
            
        #     # Open the image using cv2
        #     image = cv2.imread(image_path)
        #     if image is not None:
        #         self.database.append(self.get_VLAD(image))
        #         self.count +=1
        # print(self.count)
        # self.pre_nav_compute()
        # exit
        
    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    # @profile
    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    self.snap()
                if event.key == pygame.K_1:
                    self.map_left()
                    return Action.LEFT
                if event.key == pygame.K_2:
                    self.map_right()
                    return Action.RIGHT
                
                # Check if the pressed key is in the keymap
                if event.key == pygame.K_w:
                    self.shift_minimap('up', shift_amt)
                elif event.key == pygame.K_a:
                    self.shift_minimap('left', shift_amt)
                elif event.key == pygame.K_s:
                    self.shift_minimap('down', shift_amt)
                elif event.key == pygame.K_d:
                    self.shift_minimap('right', shift_amt)
                if event.key == pygame.K_i:
                    self.camera_position[1] -= 1
                    self.update_character_on_layer()
                elif event.key == pygame.K_j:
                    self.camera_position[0] -= 1
                    self.update_character_on_layer()
                elif event.key == pygame.K_k:
                    self.camera_position[1] += 1
                    self.update_character_on_layer()
                elif event.key == pygame.K_l:
                    self.camera_position[0] += 1
                    self.update_character_on_layer()
                    
                
                if event.key == pygame.K_c:
                    self.camera_direction = np.float64(0)
                    self.camera_position = np.array([500, 500], dtype=np.float64)
                    self.update_character_on_layer()
                    self.minimap.fill(UNEXPLORED)
                    self.wall_minimap.fill((0,0,0,0))
                if event.key == pygame.K_b:
                    self.show_target_images()
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                # else:
                    # If a key is pressed that is not mapped to an action, then display target images
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
                    
        turn = False
        if self.last_act & Action.LEFT:
            turn = True
            self.map_left()
        if self.last_act & Action.RIGHT:
            turn = True
            self.map_right()
        if not turn and (self.last_act & Action.FORWARD) and not (self.at_wall and self.notice_walls):
            self.map_forward()
        if self.last_act & Action.BACKWARD:
            self.map_backward()
        if turn or (self.at_wall and self.notice_walls):
            return self.last_act & ~Action.FORWARD
        else:
            return self.last_act
    
    def snap(self):
        nearest_90 = round(self.camera_direction / 90) * 90
        self.camera_direction = nearest_90 % 360
        self.update_character_on_layer()

    def map_forward(self):
        self.check_map()
        rad_angle = np.deg2rad(self.camera_direction)
        movement_vector = np.array([np.sin(rad_angle), -np.cos(rad_angle)]) * self.movement_step
        self.camera_position += movement_vector
        self.update_character_on_layer()
        if self._state[1] == Phase.EXPLORATION:
            draw_fov(self.minimap, self.camera_position, self.camera_direction, self.fov_radius) # This is ugly perhaps replace with circle thing
        
        
    def map_backward(self):
        self.check_map()
        rad_angle = np.deg2rad(self.camera_direction)
        movement_vector = np.array([np.sin(rad_angle), -np.cos(rad_angle)]) * self.movement_step
        self.camera_position -= movement_vector
        self.update_character_on_layer()
        if self._state[1] == Phase.EXPLORATION:
            draw_fov(self.minimap, self.camera_position, self.camera_direction, self.fov_radius)
        # pygame.draw.circle(self.minimap, FREE, (int(self.camera_position[0]), int(self.camera_position[1])), explore_radius)
        # self.occupancy_grid[int(self.camera_position[1]), int(self.camera_position[0])] = 1

    def update_character_on_layer(self):
        """ Update the arrow direction on a separate layer. """
        self.character_layer.fill((0, 0, 0, 0))  # Clear previous drawing
        
        # Calculate and draw the arrow
        arrow_points = self.calculate_arrow(self.camera_position[0], self.camera_position[1], self.camera_direction)
        pygame.draw.polygon(self.character_layer, (255, 0, 0), arrow_points)  # Draw a red arrow

    def map_left(self):
        self.camera_direction = (self.camera_direction - self.rotation_step) % 360
        self.update_character_on_layer()
        
    def map_right(self):
        self.camera_direction = (self.camera_direction + self.rotation_step) % 360
        self.update_character_on_layer()
        
    def show_target_images(self):
        # print("in function show target images")
        self.targets = self.get_target_images()

        if  self.targets is None or len( self.targets) == 0:
            return

        # self.vertical_strips = []

        for target in  self.targets:
            img_strip = target

            self.vertical_strips.append(img_strip)

        # final_display = cv2.hconcat(self.vertical_strips)
        # cv2.imshow('KeyboardPlayer: Target Images with Neighbors', final_display)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return 
        # pass 
    
    def show_pane(self):
        try:
            print("showing pane")
            loop = True
            ids = []
            key = 0
            targets = self.get_target_images()
            num_neightbors = 2 
            while loop:
                loop = False

                if targets is None or len(targets) == 0:
                    return

                vertical_strips = []

                descs = ['Front', 'Back', 'Left', 'Right']
                
                
                for (i, target) in enumerate(targets):
                    neighbor_indices = self.get_neighbor(target, count=2)
                    # print(neighbor_indices)
                    neighbors = [cv2.imread(f"{self.save_dir}{idx}.jpg") for idx in neighbor_indices]

                    cv2.putText(target, str(descs[i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    img_strip = target

                    for (j, neighbor) in enumerate(neighbors):
                        cv2.putText(neighbor, str((i*2)+j), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        img_strip = cv2.vconcat([img_strip, neighbor])
                        ids.append(neighbor_indices[j])
                    vertical_strips.append(img_strip)

                final_display = cv2.hconcat(vertical_strips)

                cv2.imshow('KeyboardPlayer: Target Images with Neighbors', final_display)
                key = cv2.waitKey(0)
                if key - 48 > 9: loop = True
            
            # self.path_layer = pygame.Surface(INITIAL_MINIMAP_SIZE, pygame.SRCALPHA)
            # self.path_layer.fill((0,0,0,0))
            
            neighbor_index = ids[key - 48]
            # print("neighbor_index=", neighbor_index)
            # print("len(self.coords)=", len(self.coords))
            # print(self.coords[neighbor_index])
            start_coords = self.camera_position[0], self.camera_position[1]
            end_coords = self.coords[neighbor_index][0] + self.transform[0], self.coords[neighbor_index][1] + self.transform[1]
            
            self.path_layer = pygame.Surface(INITIAL_MINIMAP_SIZE, pygame.SRCALPHA)
            self.path_layer.fill((0,0,0,0))
            
            pygame.draw.circle(self.path_layer, (0, 255, 0), start_coords, 5)
            pygame.draw.circle(self.path_layer, (255, 0, 0), end_coords, 5)
            
            
            # wall_minimap_opencv = pygame_surface_to_opencv(self.wall_minimap)
            # minimap_opencv = pygame_surface_to_opencv(self.minimap)
            # combined_image = overlay_images(minimap_opencv, wall_minimap_opencv)
            # cv2.imshow('Overlayed Image', combined_image)
            # cv2.waitKey(0)

            # process_and_draw_path(combined_image, start_coords, end_coords)
            
            cv2.destroyAllWindows()
            self.show_target_images()
                
        except:
            print("ERROR")


    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def compute_sift_features(self):
        """
        Compute SIFT features for images in the data directory
        """
        length = len(os.listdir(self.save_dir))
        sift_descriptors = list()
        for i in range(length):
            path = str(i) + ".jpg"
            img = cv2.imread(os.path.join(self.save_dir, path))
            # Pass the image to sift detector and get keypoints + descriptions
            # We only need the descriptors
            # These descriptors represent local features extracted from the image.
            _, des = self.sift.detectAndCompute(img, None)
            # Extend the sift_descriptors list with descriptors of the current image
            sift_descriptors.extend(des)
        return np.asarray(sift_descriptors)
    
    def get_VLAD(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        _, des = self.sift.detectAndCompute(img, None)
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_
        # Get the number of clusters from the codebook
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        # Loop over the clusters
        for i in range(k):
            # If the current cluster label matches the predicted one
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies 
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noice and variations in illumination which helps improve the 
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def get_neighbor(self, img, count):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        _, index = self.tree.query(q_VLAD, count)
        return index[0]

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # print("count: ", self.count)

        # If this function is called after the game has started
        if self.count > 0:
            # below 3 code lines to be run only once to generate the codebook
            # Compute sift features for images in the database

            # sift_descriptors = self.compute_sift_features()

            # KMeans clustering algorithm is used to create a visual vocabulary, also known as a codebook,
            # from the computed SIFT descriptors.
            # n_clusters = 64: Specifies the number of clusters (visual words) to be created in the codebook. In this case, 64 clusters are being used.
            # init='k-means++': This specifies the method for initializing centroids. 'k-means++' is a smart initialization technique that selects initial 
            # cluster centers in a way that speeds up convergence.
            # n_init=10: Specifies the number of times the KMeans algorithm will be run with different initial centroid seeds. The final result will be 
            # the best output of n_init consecutive runs in terms of inertia (sum of squared distances).
            # The fit() method of KMeans is then called with sift_descriptors as input data. 
            # This fits the KMeans model to the SIFT descriptors, clustering them into n_clusters clusters based on their feature vectors

            # TODO: try tuning the function parameters for better performance
            # codebook = KMeans(n_clusters = 64, init='k-means++', n_init=10, verbose=1).fit(sift_descriptors)
            # pickle.dump(codebook, open("codebook.pkl", "wb"))

            # Build a BallTree for fast nearest neighbor search
            # We create this tree to efficiently perform nearest neighbor searches later on which will help us navigate and reach the target location

            # TODO: try tuning the leaf size for better performance
            tree = BallTree(self.database, leaf_size=4)
            self.tree = tree


            original_position = np.array([500, 500, 1])
            new_camera_position = original_position[0] + self.transform[0], original_position[1] + self.transform[1]
            self.camera_position = new_camera_position
            self.camera_direction = 0
        
            self.show_pane()
        
        
        self.update_character_on_layer()
            

    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """

        # TODO: could you write this function in a smarter way to not simply display the image that closely 
        # matches the current FPV but the image that can efficiently help you reach the target?

        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        index = self.get_neighbor(self.fpv)
        # Display the image 5 frames ahead of the neighbor, so that next best view is not exactly same as current FPV
        self.display_img_from_id(index+5, f'Next Best View')
        # Display the next best view id along with the goal id to understand how close/far we are from the goal
        print(f'Next View ID: {index+5} || Goal ID: {self.goal}')

    # @profile
    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w * 2 + INITIAL_MINIMAP_SIZE[0], max(h, INITIAL_MINIMAP_SIZE[1])))
            

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?

                # Get full absolute save path
                save_dir_full = os.path.join(os.getcwd(),self.save_dir)
                save_path = save_dir_full + str(self.count) + ".jpg"
                # Create path if it does not exist
                if not os.path.isdir(save_dir_full):
                    os.mkdir(save_dir_full)
                # Save current FPV
                cv2.imwrite(save_path, fpv)

                # Get VLAD embedding for current FPV and add it to the database
                VLAD = self.get_VLAD(self.fpv)
                self.database.append(VLAD)
                global_coords = [self.camera_position[0] - self.transform[0], self.camera_position[1] - self.transform[1]]
                self.coords.append(global_coords)
                self.count +=1 
                self.wall_points = map_walls(fpv, wall_size_threshold)
                self.wall_points = transform_points(self.wall_points, self.camera_position, self.camera_direction, self.wall_scale)
                self.draw_walls()


            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()

        
        self.at_wall = radar_mask(fpv)
        
        resized_img = cv2.resize(fpv, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        rgb = convert_opencv_img_to_pygame(resized_img)
        self.screen.blit(rgb, (0, 0))
            
        self.screen.blit(self.minimap, (640, 0))
        self.screen.blit(self.wall_minimap, (640, 0))
        
        if self.path_layer is not None:
            self.screen.blit(self.path_layer, (640, 0))
            
        self.screen.blit(self.character_layer, (640, 0))
        
        pygame.display.update()

    def draw_walls(self):
        wall_points = np.array(self.wall_points, dtype=int)
        for x, y in wall_points:
            pygame.draw.circle(self.wall_minimap, OBSTACLE, (x, y), self.wall_circle_size)  # Draw walls


    def calculate_arrow(self, x, y, angle, length=15, width=5):
        rad_angle = np.deg2rad(angle)
        end_point_x = x + int(length * np.sin(rad_angle))
        end_point_y = y - int(length * np.cos(rad_angle))

        left_base_x = x + int(width * np.cos(rad_angle))
        left_base_y = y + int(width * np.sin(rad_angle))
        right_base_x = x - int(width * np.cos(rad_angle))
        right_base_y = y - int(width * np.sin(rad_angle))

        return [(end_point_x, end_point_y), (left_base_x, left_base_y), (right_base_x, right_base_y)]

    def apply_transform(self, position):
        homogeneous_position = np.array([position[0], position[1], 1])
        transformed = np.dot(self.transform, homogeneous_position)
        return transformed[:2]

    def apply_inverse_transform(self, position):
        try:
            inverse_transform = np.linalg.inv(self.transform)
            homogeneous_position = np.array([position[0], position[1], 1])
            transformed = np.dot(inverse_transform, homogeneous_position)
            return transformed[:2]
        except np.linalg.LinAlgError:
            return np.array([position[0], position[1]])

    def check_map(self):
        width, height = self.minimap.get_size()
        camera_x, camera_y = self.camera_position
        if (camera_x <= self.scale_threshold): self.shift_minimap('right', self.scale_threshold*3)
        elif (camera_y <= self.scale_threshold): self.shift_minimap('down', self.scale_threshold*3)
        elif (camera_y >= height - self.scale_threshold): self.shift_minimap('up', self.scale_threshold*3)
        elif (camera_x >= width - self.scale_threshold): self.shift_minimap('left', self.scale_threshold*3)
        
    
    def shift_minimap(self, direction, pixels):
        shift_x = 0
        shift_y = 0

        if direction == 'up':
            shift_y = -pixels
        elif direction == 'down':
            shift_y = pixels
        elif direction == 'left':
            shift_x = -pixels
        elif direction == 'right':
            shift_x = pixels

        self.camera_position[0] += shift_x
        self.camera_position[1] += shift_y
        
        self.transform[0] += shift_x
        self.transform[1] += shift_y
        
        new_width, new_height = self.minimap.get_size()
        new_minimap = pygame.Surface((new_width, new_height))
        new_wall_minimap = pygame.Surface((new_width, new_height), pygame.SRCALPHA)
        new_character_layer = pygame.Surface((new_width, new_height), pygame.SRCALPHA)

        new_minimap.fill(UNEXPLORED)
        new_wall_minimap.fill((0, 0, 0, 0))
        new_character_layer.fill((0, 0, 0, 0))

        new_minimap.blit(self.minimap, (shift_x, shift_y))
        new_wall_minimap.blit(self.wall_minimap, (shift_x, shift_y))
        new_character_layer.blit(self.character_layer, (shift_x, shift_y))

        self.minimap = new_minimap
        self.wall_minimap = new_wall_minimap
        self.character_layer = new_character_layer

            
if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
