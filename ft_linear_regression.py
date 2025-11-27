import shared
import train
from pathlib import Path
import csv
import numpy as np


def main():
    file_csv = "data.csv"
    while True:
        print()
        try:
            if Path("./thetas.csv").is_file() and shared.theta0 == 0 and shared.theta1 == 0:
                user_input = input("thetas.csv found. do you wanna load values ?\n\
(y or yes / n or no / q or quit)\n")
                if user_input == 'y' or user_input == "yes":
                    with open("thetas.csv", 'r', newline='') as csvF:
                        csv_file = csv.reader(csvF) 
                        try:
                            first_line = next(csv_file)
                        except StopIteration:
                            raise ValueError(f"file is empty.")

                    if len(first_line) < 2:
                        raise ValueError("need 2 values on first line")

                    try:
                        shared.theta0 = float(first_line[0])
                        shared.theta1 = float(first_line[1])
                        shared.data = np.loadtxt("thetas.csv",
                                     delimiter=',',
                                     skiprows=1)
                        
                    except ValueError:
                        raise ValueError("values are invalid")

                elif user_input == 'n' or user_input == "no":
                    pass
                elif user_input == 'q' or user_input == "quit":
                    quit()
                else:
                    print("hahaha ! so funny")

        # 1 check if thetas are initialized (meaning a model has been trained)
            if shared.theta0 == 0 and shared.theta1 == 0:
                user_input = input("theta0 and theta1 are not initialized, \
do you wanna train the model with data ? (y | n) > ").lower()

            # YES case ----------------------------------------------------
                if user_input == 'y' or user_input == "yes":
                    path = Path("data.csv")
                    if path.exists() and path.is_file():
                        train.main(file_csv)
                    else:
                        print("data.csv is missing")

            # NO case ----------------------------------------------------
                elif user_input == 'n' or user_input == "no":
                    user_input = input("train with an other model ? \
(file/path | n) > ")
                    if user_input == 'n' or user_input == "no":
                        print("all estimations are 0 ¯\_(ツ)_/¯")
                    elif user_input == "quit" or user_input == "q":
                        break
                    else:
                        path = Path(user_input)
                        if path.exists() and path.is_file():
                            file_csv = user_input
                            train.main(file_csv)
                        else:
                            print(f"file {user_input} not found")

            # QUIT case ----------------------------------------------------
                elif user_input == "quit" or user_input == "q":
                    break

            # NUM case ----------------------------------------------------
                elif user_input.isdigit():
                    print(0)

            # BS case ----------------------------------------------------
                else:
                    print("Hahaha ! So funny !")

        # theta0 and thetha1 initialize (a model has been trained)
            else:
                user_input = input("enter a value for model estimation\n\
(q or quit to leave / t or train to train an other model / s or stats for model stats \
sa or save for saving values / print or p for printing values on terminal)\n> ")

            # QUIT case ----------------------------------------------------
                if user_input == "quit" or user_input == "q":
                    break

            # TRAIN new model case ------------------------------------------
                elif user_input == "train" or user_input == "t":
                    user_input = input("train with an other model ? \
(file/path | n) > ")
                    if user_input == "quit" or user_input == "q":
                        break
                    elif user_input == 'n' or user_input == "no":
                        pass
                    else:
                        path = Path(user_input)
                        if path.exists() and path.is_file():
                            file_csv = user_input
                            train.main(file_csv)
                        else:
                            print(f"file {user_input} not found")

            # STATS case ----------------------------------------------------
                elif user_input == "stats" or user_input == 's':
                    x_data = shared.data[:, 0]
                    y_true = shared.data[:, 1]
                    y_pred = shared.theta0 * x_data + shared.theta1
                    print(f"""
Mean Absolute Error = {train.calculate_mae(y_true, y_pred)}
Root Mean Square Error = {train.calculate_rmse(y_true, y_pred)}
Model accuracy (r2) = {train.calculate_r2(y_true, y_pred)} (coef de determination)
                      """)

                elif user_input == "save" or user_input == 'sa':
                    with open("./thetas.csv", 'w', newline='', encoding='utf-8') as saveF:
                        writer = csv.writer(saveF)
                        writer.writerows([[shared.theta0, shared.theta1]])
                        try:
                            np.savetxt(saveF,               # Passe l'objet fichier ouvert
                                       shared.data,         # Le tableau NumPy à sauvegarder
                                       delimiter=",",
                                       fmt='%.17f')         # Assure une bonne précision des flottants
                            print("Coefficients et données d'entraînement sauvegardés dans thetas.csv.")
                        except AttributeError:
                            print("Erreur : shared.data n'est pas un tableau NumPy valide. Sauvegarde des données annulée.")
                        except Exception as e:
                            print(f"Erreur lors de la sauvegarde des données : {e}")

                elif user_input == "print" or user_input == 'p':
                    print("theta0 = ", shared.theta0)
                    print("theta1 = ", shared.theta1)
                    print()
                    print(shared.data)

            # BASIC case (answering value user input by model estimation)
                else:
                    kmtrage = float(user_input)
                    if kmtrage < 0:
                        print("negative input detected")
                    estimation = shared.theta0 * kmtrage + shared.theta1
                    if estimation < 0:
                        print("negative result detected")
                    print(f"estimated price = {estimation}")

    # throws
        except EOFError:
            print()
            break
        except ValueError:
            print("invalid input, please enter a number")
        except KeyboardInterrupt:
            print()
            quit()


if __name__ == "__main__":
    main()
