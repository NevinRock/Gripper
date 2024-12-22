import pybullet as p
import time
import random
from abc import ABC, abstractmethod
import pybullet_data
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import lightgbm as lgb
from sklearn.metrics import ConfusionMatrixDisplay


class Gripper(ABC):
    def __init__(self):
        self.obj_pos = [0, 0, 0]
        self.obj_ori = p.getQuaternionFromEuler([0, 0, 0])

    @abstractmethod
    def grasp(self):
        """
        The function that controls the claw's grip on an object
        """

        pass

    @abstractmethod
    def openGripper(self):
        """
        The function to open the gripper
        """
        pass

    @abstractmethod
    def preshape(self):
        """
        Claw initialization
        """
        pass

    @abstractmethod
    def run(self):
        """
        main function to generate training data
        """
        pass

    @abstractmethod
    def generate_claw_positions_and_orientations(self):
        """
        generate random position
        """
        pass

    # ploting functinos
    # Function to plot scatter plot of positions
    @staticmethod
    def plot_scatter_positions(df):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Extracting success and unsuccess points
        success_points = df[df["Success"] == 1]
        failure_points = df[df["Success"] == 0]

        # Green color marking for success points and plot the scatter plot
        ax.scatter(
            [pos[0] for pos in success_points["Position"]],
            [pos[1] for pos in success_points["Position"]],
            [pos[2] for pos in success_points["Position"]],
            c="green", label="Success", alpha=0.6, s=5
        )

        # Red color marking for unsuccess points and plot in scatter
        ax.scatter(
            [pos[0] for pos in failure_points["Position"]],
            [pos[1] for pos in failure_points["Position"]],
            [pos[2] for pos in failure_points["Position"]],
            c="red", label="Failure", alpha=0.6, s=5
        )

        # Setting up the labels,title and legend
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("Scatter Plot of Grasp Positions")
        ax.legend()
        plt.show()

    def plot_scatter_with_orientation(self, df):
        """
        Function to plot scatter plot with orientation
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot all points as positions and arrows to represent the orientations
        for _, row in df.iterrows():
            position = row["Position"]
            orientation = row["Orientation"]
            success = row["Success"]

            # Convert quaternion to a direction vector
            r = R.from_quat(orientation)

            # orientation points along the z-axis for Three-finger gripper and
            # along x-axis for Two-finger
            if self.num_finger == 2:
                direction = r.apply([1, 0, 0])
            elif self.num_finger == 3:
                direction = r.apply([0, 0, 1])

            # Plot the position (same color marker as the position scatter)
            color = "green" if success == 1 else "red"
            ax.scatter(
                position[0],
                position[1],
                position[2],
                c=color,
                alpha=0.6,
                s=5)

            # Plot the orientation as an arrow
            ax.quiver(position[0], position[1], position[2],  # Start point
                      # Direction vector
                      direction[0], direction[1], direction[2],
                      color=color, alpha=0.6, length=0.2, normalize=True)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("Scatter Plot of Grasp Positions with Orientation")
        plt.show()

    @staticmethod
    def plot_kde_heatmap(df):
        """
        Function to plot KDE heatmap
        """

        # Convert the x,y,z position to lists
        positions = np.array(df["Position"].tolist())
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Plot KDE on 3D positions
        xyz = np.vstack([x, y, z])
        kde = gaussian_kde(xyz)

        # Create the grid
        grid_size = 50
        x_grid = np.linspace(x.min(), x.max(), grid_size)
        y_grid = np.linspace(y.min(), y.max(), grid_size)
        z_grid = np.linspace(z.min(), z.max(), grid_size)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Evaluate the KDE
        kde_values = kde(grid_points).reshape(grid_size, grid_size, grid_size)

        # Plot the heatmap as contour slices
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.contourf(xx[:, :, 0], yy[:, :, 0], kde_values[:, :,
                    int(grid_size / 2)], cmap="hot", alpha=0.6)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("KDE Heatmap of Grasp Positions")
        plt.show()

    @staticmethod
    def read_all_csv_files(directory):
        """
        put all csv files in the directory into a list
        """

        all_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
        if not all_files:
            # error message if we can't find any csv files in the directory
            raise FileNotFoundError(
                f"No CSV files found in directory '{directory}'.")
        combined_df = pd.concat([pd.read_csv(os.path.join(directory, f))
                                for f in all_files], ignore_index=True)
        return combined_df

    def is_object_grasped(self, object_id):
        """
        contact point between hand and obj
        """
        contact_points = p.getContactPoints(
            bodyA=self.robot_model, bodyB=object_id)
        if len(contact_points) > 0:
            # contact force
            for point in contact_points:
                if point[9] > 1e-3:  # normal force
                    print("Successful grasp")
                    return True
        print("Unsuccess grasp")
        return False

    def gene_noise(self):
        """
        generate the Gussian noise for orientation and position
        """

        # Define the Gaussian noise parameters
        position_noise_std = 0.002  # position noise (2mm)
        # orientation noise (around 10 degrees)
        orientation_noise_std = np.radians(0.175)

        # Add Gaussian noise to position
        x = np.random.normal(0, position_noise_std)
        y = np.random.normal(0, position_noise_std)
        z = np.random.normal(0, position_noise_std)

        # Add Gaussian noise to orientation
        x_roll = np.random.normal(0, orientation_noise_std)
        y_roll = np.random.normal(0, orientation_noise_std)
        z_roll = np.random.normal(0, orientation_noise_std)

        pos_noise = np.array([x, y, z])
        ori_noise = np.array([x_roll, y_roll, z_roll])

        return pos_noise, ori_noise

    def PCA(self, directory):
        """
        PCA function
        """

        # read the data
        df = self.read_all_csv_files(directory)

        # unzip the data
        df['Position'] = df['Position'].apply(ast.literal_eval)
        df['Orientation'] = df['Orientation'].apply(ast.literal_eval)

        position_df = pd.DataFrame(
            df['Position'].tolist(), columns=[
                'Pos_X', 'Pos_Y', 'Pos_Z'])
        orientation_df = pd.DataFrame(
            df['Orientation'].tolist(), columns=[
                'Ori_W', 'Ori_X', 'Ori_Y', 'Ori_Z'])

        # conbine features
        features = pd.concat([position_df, orientation_df], axis=1)

        # standardised
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)

        # calculate variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        print(
            "Proportion of variance explained by each principal component:",
            explained_variance_ratio)
        print("Cumulative variance contribution:", cumulative_variance)

        # ploting
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(
                1,
                len(cumulative_variance) +
                1),
            cumulative_variance,
            marker='o',
            linestyle='--')
        plt.title('Cumulative variance contribution')
        plt.xlabel('Number of principal components')
        plt.ylabel('Cumulative proportion of variance')
        plt.grid()
        plt.show()

        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[
                'Principal Component 1',
                'Principal Component 2'])
        pca_df['Success'] = df['Success']

        plt.figure(figsize=(8, 6))
        for label in pca_df['Success'].unique():
            subset = pca_df[pca_df['Success'] == label]
            plt.scatter(
                subset['Principal Component 1'],
                subset['Principal Component 2'],
                label=f"Success {label}",
                edgecolor='k'
            )

        plt.title('PCA Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title="Success")
        plt.grid()
        plt.show()

    def noise_compare(self, data1_dir, data_noised_dir):
        """
        compare before and after add noise
        """
        plot_dir = "../plot_between_noise"
        os.makedirs(plot_dir, exist_ok=True)

        # Read the csv data files from the original data (data1) and the
        # increased noise data(data_noised)
        data1 = self.read_all_csv_files(data1_dir)
        data_noised = self.read_all_csv_files(data_noised_dir)

        # Preprocess data
        for df in [data1, data_noised]:
            df["Position"] = df["Position"].apply(eval)
            df["Orientation"] = df["Orientation"].apply(eval)

        # Prepare Features and Labels for data1
        X_data1 = np.hstack(
            (np.vstack(
                data1["Position"]), np.vstack(
                data1["Orientation"])))
        y_data1 = data1["Success"].values

        # Undersample before splitting (for handling of unbalanced data)
        minority_data = data1[data1["Success"] == 1]
        majority_data = data1[data1["Success"] == 0].sample(
            n=len(minority_data), random_state=42)
        data1_balanced = pd.concat([minority_data, majority_data]).sample(
            frac=1, random_state=42)

        # Split balanced dataset into Features and Labels
        X_balanced = np.hstack(
            (np.vstack(
                data1_balanced["Position"]), np.vstack(
                data1_balanced["Orientation"])))
        y_balanced = data1_balanced["Success"].values

        # Normalize the features for the balanced dataset
        scaler = MinMaxScaler()
        X_balanced_normalized = scaler.fit_transform(X_balanced)

        # Split Dataset into Train and Test Sets (22% as the test set)
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced_normalized, y_balanced, test_size=0.22, random_state=42)

        # Define the 4 classifiers
        classifiers = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "LightGBM": lgb.LGBMClassifier(objective="binary", random_state=42, n_estimators=100)
        }

        # Train classifiers on the training set of data1
        print("Training classifiers on the training set of data1...")
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)

        # Prepare Features and Labels for data_noised
        X_data_noised = np.hstack(
            (np.vstack(
                data_noised["Position"]), np.vstack(
                data_noised["Orientation"])))
        y_data_noised = data_noised["Success"].values
        X_data_noised_normalized = scaler.transform(X_data_noised)

        def evaluate_and_print(X, y, dataset_name):
            """
            Function to calculate and print the accuracy result
            """
            results = {}
            print(f"\nAccuracy for {dataset_name}:")
            for name, clf in classifiers.items():
                y_pred = clf.predict(X)
                accuracy = accuracy_score(y, y_pred)
                results[name] = accuracy
                print(f"{name}: {accuracy * 100:.2f}%")
            return results

        # test set of data1 (before noise increased)
        results_data1_test = evaluate_and_print(
            X_test, y_test, "Data before increasing noise")

        # Data after noise increased
        results_data_noised = evaluate_and_print(
            X_data_noised_normalized,
            y_data_noised,
            "Data after increasing noise")

        def plot_roc_curves(X, y, dataset_name, plot_suffix):
            """
            Plot ROC Curves
            """
            roc_curves = {}
            for name, clf in classifiers.items():
                y_score = clf.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_score)
                roc_auc = auc(fpr, tpr)
                roc_curves[name] = (fpr, tpr, roc_auc)

            plt.figure()
            for name, (fpr, tpr, roc_auc) in roc_curves.items():
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curves for {dataset_name}")
            plt.legend()
            roc_output_path = f"{plot_dir}/roc_curves_{plot_suffix}.png"
            plt.savefig(roc_output_path)
            plt.show()

        # Plot ROC Curves for data before noise increased
        plot_roc_curves(
            X_test,
            y_test,
            "Data before increasing noise",
            "data1_test")

        # Plot ROC Curves for data after noise increased
        plot_roc_curves(
            X_data_noised_normalized,
            y_data_noised,
            "Data after increasing noise",
            "data_noised")

    def train_and_plot(self, directory):
        """
        train the data and plot the graphs for results visualisation
        """
        plot_dir = "../plots"
        os.makedirs(plot_dir, exist_ok=True)

        df = self.read_all_csv_files(directory)

        # Balance the dataset for classifier training
        df["Position"] = df["Position"].apply(eval)
        df["Orientation"] = df["Orientation"].apply(eval)
        minority_data = df[df["Success"] == 1]
        majority_data = df[df["Success"] == 0].sample(
            n=len(minority_data), random_state=42)
        df_balanced = pd.concat([minority_data, majority_data]).sample(
            frac=1, random_state=42)

        # Prepare Features and Labels
        X = np.hstack(
            (np.vstack(
                df_balanced["Position"]), np.vstack(
                df_balanced["Orientation"])))
        y = df_balanced["Success"].values

        # Normalize the features
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)

        # Split Dataset into Train and Test Sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42)

        # Define Classifiers
        classifiers = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "LightGBM": lgb.LGBMClassifier(objective="binary", random_state=42, n_estimators=100)
        }

        # Classifier training and evaluation
        # Cross Validation
        cv = StratifiedKFold(n_splits=5)
        results = {}
        roc_curves = {}

        for name, clf in classifiers.items():
            print(f"Training {name}...")
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                clf.fit(X_train[train_idx], y_train[train_idx])
                cv_scores.append(clf.score(X_train[val_idx], y_train[val_idx]))
            print(f"{name} CV Accuracy: {np.mean(cv_scores) * 100:.2f}%")

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy

            print(f"\n{name} Classification Report:")
            print(classification_report(y_test, y_pred))

            # Plot Confusion Matrix
            cm_output_path = f"./{plot_dir}/confusion_matrix_{name.replace(' ', '_')}.png"
            ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
            plt.title(f"Confusion Matrix - {name}")
            plt.savefig(cm_output_path)
            plt.show()

            # ROC Curve
            y_score = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            roc_curves[name] = (fpr, tpr, roc_auc)

        # Compare Performance of Classifiers
        print("\nClassifier Performance Comparison:")
        for name, accuracy in results.items():
            print(f"{name}: {accuracy * 100:.2f}%")

        # Visualize the Comparison
        plt.bar(results.keys(), results.values())
        plt.ylabel("Accuracy")
        plt.title("Classifier Performance Comparison")
        plt.xticks(rotation=45)
        plt.savefig(f"./{plot_dir}/comparison_of_classifiers.png")
        plt.show()

        # Plot ROC Curves
        plt.figure()
        for name, (fpr, tpr, roc_auc) in roc_curves.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        roc_output_path = f"./{plot_dir}/roc_curves.png"
        plt.savefig(roc_output_path)
        plt.show()

        # Data Size vs Performance
        data_sizes = np.linspace(0.1, 1.0, 10)
        performance = {name: [] for name in classifiers.keys()}

        for size in data_sizes:
            size = int(size * len(X_train))
            X_partial = X_train[:size]
            y_partial = y_train[:size]
            for name, clf in classifiers.items():
                clf.fit(X_partial, y_partial)
                y_pred_partial = clf.predict(X_test)
                performance[name].append(
                    accuracy_score(y_test, y_pred_partial))

        # Plot Data Size vs Performance
        plt.figure()
        for name, scores in performance.items():
            plt.plot(data_sizes * len(X_train), scores, label=name)
        plt.xlabel("Training Data Size")
        plt.ylabel("Accuracy")
        plt.title("Performance vs Data Size")
        plt.legend()
        performance_output_path = f"./{plot_dir}/performance_vs_data_size.png"
        plt.savefig(performance_output_path)
        plt.show()

        # Plot scatter plots and KDE heatmap
        self.plot_scatter_positions(df)
        self.plot_scatter_with_orientation(df)
        self.plot_kde_heatmap(df)


class ThreeFinger(Gripper):
    def __init__(self):
        super().__init__()
        self.num_finger = 3

    # generate the random position and ori
    def generate_claw_positions_and_orientations(self):

        # random face
        face = random.choice(
            ['front', 'back', 'left', 'right', 'top', 'bottom'])

        half_size = 0.08  # distace for horizontal random
        dis_radius = 0.17  # distance for hand
        dis_alpha = 0.05  # hand distance random

        print(face)

        ori_alpha = 0.01  # random angle
        ori_beta = 8.5  # horizontal random angle

        # Generate pos and ori based on the face
        if face == 'front':  # face to +y  ori:[-pi/2, 0, 0]
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-dis_radius - dis_alpha, -
                               dis_radius + dis_alpha)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(-np.pi / 2 -
                                    ori_alpha, -np.pi / 2 + ori_alpha)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
        elif face == 'back':  # face to -y ori:[pi/2, 0, 0]
            x = random.uniform(-half_size, half_size)
            y = random.uniform(dis_radius - dis_alpha, dis_radius + dis_alpha)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(
                np.pi / 2 - ori_alpha,
                np.pi / 2 + ori_alpha)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
        elif face == 'left':  # face to +x ori:[0, pi/2, 0]
            x = random.uniform(-dis_radius - dis_alpha, -
                               dis_radius + dis_alpha)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(
                np.pi / 2 - ori_alpha,
                np.pi / 2 + ori_alpha)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
        elif face == 'right':  # face to -x ori:[0, -pi/2, 0]
            x = random.uniform(dis_radius - dis_alpha, dis_radius + dis_alpha)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(-np.pi / 2 -
                                    ori_alpha, -np.pi / 2 + ori_alpha)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
        elif face == 'top':  # face to -z ori:[-pi, 0, 0]
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(dis_radius - dis_alpha, dis_radius + dis_alpha)

            x_roll = random.uniform(-np.pi - ori_alpha, -np.pi + ori_alpha)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
        elif face == 'bottom':  # face to +z ori:[0, 0, 0]
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(-dis_radius - dis_alpha, -
                               dis_radius + dis_alpha)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)

        return np.array([x, y, z]), np.array([x_roll, y_roll, z_roll])

    # function to grasp object
    def grasp(self):

        done = False
        while not done:

            for i in [1, 4]:
                p.setJointMotorControl2(
                    self.robot_model,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=0.05,
                    maxVelocity=1,
                    force=1)

            p.setJointMotorControl2(
                self.robot_model,
                7,
                p.POSITION_CONTROL,
                targetPosition=0.05,
                maxVelocity=1,
                force=2)
            done = True
        self.open = False

    def preshape(self):

        done = False
        while not done:

            for i in [2, 5, 8]:
                p.setJointMotorControl2(
                    self.robot_model,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=0.4,
                    maxVelocity=2,
                    force=1)

            done = True
        self.open = False

    def openGripper(self):
        closed = True
        iteration = 0
        while (closed and not self.open):
            joints = self.getJointPosition()
            closed = False
            for k in range(0, self.numJoints):

                # lower finger joints
                if k == 2 or k == 5 or k == 8:
                    goal = 0.9
                    if joints[k] >= goal:
                        p.setJointMotorControl2(
                            self.robot_model,
                            k,
                            p.POSITION_CONTROL,
                            targetPosition=joints[k] -
                            0.05,
                            maxVelocity=2,
                            force=5)
                        closed = True

                        # Upper finger joints
                elif k == 6 or k == 3 or k == 9:
                    goal = 0.9
                    if joints[k] <= goal:
                        p.setJointMotorControl2(
                            self.robot_model,
                            k,
                            p.POSITION_CONTROL,
                            targetPosition=joints[k] -
                            0.05,
                            maxVelocity=2,
                            force=5)
                        closed = True

                        # Base finger joints
                elif k == 1 or k == 4 or k == 7:
                    pos = 0.9
                    if joints[k] <= pos:
                        p.setJointMotorControl2(
                            self.robot_model,
                            k,
                            p.POSITION_CONTROL,
                            targetPosition=joints[k] -
                            0.05,
                            maxVelocity=2,
                            force=5)
                        closed = True

            iteration += 1
            if iteration > 10000:
                break
            p.stepSimulation()
        self.open = True

    def getJointPosition(self):
        joints = []
        for i in range(0, self.numJoints):
            joints.append(p.getJointState(self.robot_model, i)[0])
        return joints

    # init the finger position
    def set_init_finger_pose(self):
        self.reset_finger([[0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0.7, 0.8, 0.8, 0.8]])

    def reset_finger(self, hand_config):
        for i, finger_index in enumerate(self.fingers):
            for j, joint in enumerate(finger_index):
                p.resetJointState(self.robot_model, joint, hand_config[i][j])

    # main run function
    def run(self):

        p.setRealTimeSimulation(1)  # 1 real time, 0 no real time

        # init the hand
        path = "../model/threeFingers/sdh.urdf"

        # gene the random hand ori and pos
        self.hand_pos, self.hand_ori = self.generate_claw_positions_and_orientations()

        # transfer hand pos
        self.hand_pos += np.array(self.obj_pos)

        # add noise
        pos_noise, ori_noise = self.gene_noise()
        self.hand_pos += pos_noise
        self.hand_ori += ori_noise

        # transfer hand ori
        self.hand_ori = p.getQuaternionFromEuler(self.hand_ori)
        # load hand model
        self.robot_model = p.loadURDF(path, self.hand_pos, self.hand_ori,
                                      globalScaling=1, useFixedBase=False)

        self.hand_base_controller = p.createConstraint(self.robot_model,
                                                       -1,
                                                       -1,
                                                       -1,
                                                       p.JOINT_FIXED,
                                                       [0, 0, 0], [0, 0, 0], self.hand_ori)
        p.changeConstraint(
            self.hand_base_controller,
            self.hand_pos,
            jointChildFrameOrientation=self.hand_ori,
            maxForce=50)

        # lookup the parameters of this!
        p.addUserDebugLine([0, 0, 0], [0.5, 0, 0], [1, 0, 0], 1, 0)
        p.addUserDebugLine([0, 0, 0], [0, 0.5, 0], [0, 1, 0], 1, 0)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.5], [0, 0, 1], 1, 0)

        self.open = False
        self.numJoints = p.getNumJoints(self.robot_model)
        self.openGripper()
        self.preshape()
        time.sleep(.05)

        # load object model
        self.objectID = p.loadURDF(
            "../model/ball.urdf",
            self.obj_pos,
            self.obj_ori,
            globalScaling=2)

        # test the pos and ori

        time.sleep(0.05)

        self.grasp()
        time.sleep(1.5)
        self.hand_pos[2] += 0.1
        p.changeConstraint(
            self.hand_base_controller,
            self.hand_pos,
            jointChildFrameOrientation=self.hand_ori,
            maxForce=50)

        time.sleep(1)

        # test graspe or not
        success = self.is_object_grasped(self.objectID)

        # delete old model
        p.removeBody(self.robot_model)
        p.removeBody(self.objectID)
        return {
            'Position': list(self.hand_pos),
            'Orientation': list(self.hand_ori),
            'Success': int(success)  # Convert True/False to 1/0
        }


class TwoFinger(Gripper):
    def __init__(self):
        super().__init__()
        self.num_finger = 2

    def generate_claw_positions_and_orientations(self):
        # random face
        face = random.choice(
            ['front', 'back', 'left', 'right', 'top', 'bottom'])

        half_size = 0.08
        dis_radius = 0.465
        dis_alpha = 0.05

        print(face)

        ori_beta = 20
        ori_alpha = -np.pi / ori_beta

        # gene pos and ori
        if face == 'front':  # face to +y  ori:[0,0,pi/2]

            x = random.uniform(-half_size, half_size)
            y = random.uniform(-dis_radius - dis_alpha, -
                               dis_radius + dis_alpha)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(
                np.pi / 2 - ori_alpha,
                np.pi / 2 + ori_alpha)
        elif face == 'back':  # face to -y ori:[0, 0, -pi/2]

            x = random.uniform(-half_size, half_size)
            y = random.uniform(dis_radius - dis_alpha, dis_radius + dis_alpha)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(-np.pi / 2 -
                                    ori_alpha, -np.pi / 2 + ori_alpha)
        elif face == 'left':  # face to +x ori:[0, 0, 0]

            x = random.uniform(-dis_radius - dis_alpha, -
                               dis_radius + dis_alpha)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
        elif face == 'right':  # face to -x ori:[0, 0, pi]

            x = random.uniform(dis_radius - dis_alpha, dis_radius + dis_alpha)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(-half_size, half_size)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            z_roll = random.uniform(np.pi - ori_alpha, np.pi + ori_alpha)
        elif face == 'top':  # face to -z ori:[0, pi/2, 0]

            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(dis_radius - dis_alpha, dis_radius + dis_alpha)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(
                np.pi / 2 - ori_alpha,
                np.pi / 2 + ori_alpha)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
        elif face == 'bottom':  # face to +z ori:[0, -pi/2, 0]

            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            z = random.uniform(-dis_radius - dis_alpha, -
                               dis_radius + dis_alpha)

            x_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)
            y_roll = random.uniform(-np.pi / 2 -
                                    ori_alpha, -np.pi / 2 + ori_alpha)
            z_roll = random.uniform(-np.pi / ori_beta, np.pi / ori_beta)

        return np.array([x, y, z]), np.array([x_roll, y_roll, z_roll])

    def grasp(self):
        p.setJointMotorControl2(self.robot_model, 0, p.POSITION_CONTROL,
                                targetPosition=0, maxVelocity=2, force=10)
        p.setJointMotorControl2(self.robot_model, 2, p.POSITION_CONTROL,
                                targetPosition=0, maxVelocity=2, force=10)

    def openGripper(self):
        p.setJointMotorControl2(self.robot_model, 0, p.POSITION_CONTROL,
                                targetPosition=4, maxVelocity=2, force=10)
        p.setJointMotorControl2(self.robot_model, 2, p.POSITION_CONTROL,
                                targetPosition=4, maxVelocity=2, force=10)

    def preshape(self):
        p.setJointMotorControl2(self.robot_model, 0, p.POSITION_CONTROL,
                                targetPosition=4, maxVelocity=2, force=1)
        p.setJointMotorControl2(self.robot_model, 2, p.POSITION_CONTROL,
                                targetPosition=4, maxVelocity=2, force=1)

    def run(self):
        p.setRealTimeSimulation(1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # gene random  pos and ori
        self.hand_pos, self.hand_ori = self.generate_claw_positions_and_orientations()

        # transfer hand pos
        self.hand_pos += np.array(self.obj_pos)

        # add noise
        pos_noise, ori_noise = self.gene_noise()
        self.hand_pos += pos_noise

        self.hand_ori += ori_noise
        self.hand_ori = p.getQuaternionFromEuler(self.hand_ori)

        # load hand model
        self.robot_model = p.loadURDF(
            "pr2_gripper.urdf",
            self.hand_pos,
            self.hand_ori,
            globalScaling=1.7,
            useFixedBase=False)

        self.hand_base_controller = p.createConstraint(self.robot_model,
                                                       -1, -1, -1, p.JOINT_FIXED,
                                                       [0, 0, 0], [0, 0, 0], self.hand_ori)

        p.changeConstraint(
            self.hand_base_controller,
            self.hand_pos,
            jointChildFrameOrientation=self.hand_ori,
            maxForce=50)

        self.numJoints = p.getNumJoints(self.robot_model)
        self.openGripper()
        self.preshape()
        time.sleep(0.5)

        # load object moddel
        self.objectID = p.loadURDF(
            "../model/ball.urdf",
            self.obj_pos,
            self.obj_ori,
            globalScaling=2)

        # test hand
        self.grasp()
        time.sleep(1.5)
        self.hand_pos[2] += 0.3
        p.changeConstraint(
            self.hand_base_controller,
            self.hand_pos,
            jointChildFrameOrientation=self.hand_ori,
            maxForce=50)

        time.sleep(1)
        success = self.is_object_grasped(self.objectID)
        # delete the old objects
        p.removeBody(self.robot_model)
        p.removeBody(self.objectID)
        return {
            'Position': list(self.hand_pos),
            'Orientation': list(self.hand_ori),
            'Success': int(success)  # Convert True/False to 1/0
        }


if __name__ == '__main__':

    obj = ThreeFinger()

    # ask for traning
    train_ask = input("Do you want to generate new data? (y/n) ")

    if train_ask == 'y':

        train_time_ask = int(input("How many data you want to generate"))

        # gene the data
        clid = p.connect(p.SHARED_MEMORY)
        if (clid < 0):
            p.connect(p.GUI)

        obj = ThreeFinger()
        # Store results
        results = []
        batch_size = 200  # Number of data entries per CSV file
        file_counter = 1  # Counter for file names

        for i in range(train_time_ask):  # Total trials
            try:
                print(f"Running trial {i + 1}...")
                result = obj.run()
                results.append(result)

                # Save results to a new file every 200 trials
                if len(results) == batch_size:
                    # Save to CSV
                    df = pd.DataFrame(results)
                    file_name = f"grasp_results{file_counter}.csv"
                    df.to_csv(file_name, index=False)
                    print(f"Batch saved to {file_name}")

                    # Clear results for the next batch and increment file
                    # counter
                    results = []
                    file_counter += 1

            except Exception as e:
                print(f"Error during trial {i + 1}: {e}")

        # Save any remaining results after the loop
        if results:
            df = pd.DataFrame(results)
            file_name = f"grasp_results{file_counter}.csv"
            df.to_csv(file_name, index=False)
            print(f"Final batch saved to {file_name}")

        # Disconnect PyBullet
        p.disconnect()

        # plot the graph

        obj.PCA('./')
        obj.train_and_plot('./')

    elif train_ask == 'n':

        obj.PCA('../data/three_finger_ball_data')
        obj.train_and_plot('../data/three_finger_ball_data')
        obj.noise_compare(
            '../data/three_finger_ball_data',
            '../data/three_finger_ball_data_noise')
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
