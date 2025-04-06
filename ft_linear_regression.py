import shared
import train
from pathlib import Path


def main():
    while True:
        try:
            # if no model are train
            if shared.theta0 == 0 and shared.theta1 == 0:
                user_input = input("theta0 and theta1 are not initialized, \
do you wanna train the model with data ? (y | n) > ").lower()

                # if yes for subject basic model trainning
                if user_input == 'y' or user_input == "yes":
                    train.main("data.csv")

                # else ask if wanna train an other model
                elif user_input == 'n' or user_input == "no":
                    user_input = input("train with an other model ? \
(file/path | n) > ")
                    # if no, then noting
                    if user_input == 'n' or user_input == "no":
                        print("all estimations are 0 ¯\_(ツ)_/¯")
                    # else check if file exist and tain with it
                    elif user_input == "quit" or user_input == "q":
                        break
                    else:
                        path = Path(user_input)
                        if path.exists() and path.is_file():
                            print("hello world")
                            train.main(user_input)
                        else:
                            print(f"file {user_input} not found")

                # quit and BS entry
                elif user_input == "quit" or user_input == "q":
                    break

                else:
                    print("hahaha so funny !")

            # else si a model has been trained
            else:
                user_input = input("enter a value\n\
(q or quit to leave / t or train to train an other model / s or stats for model stats)\n> ")

                if user_input == "quit" or user_input == "q":
                    break

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
                            train.main(user_input)
                        else:
                            print(f"file {user_input} not found")

                elif user_input == "stats" or user_input == 's':
                    x_data = shared.data[:, 0]
                    y_true = shared.data[:, 1]
                    y_pred = shared.theta0 * x_data + shared.theta1
                    print(f"""
Mean Absolute Error = {train.calculate_mae(y_true, y_pred)} 
Root Mean Square Error = {train.calculate_rmse(y_true, y_pred)}
Model accuracy (r2) = {train.calculate_r2(y_true, y_pred)} (coef de determination)
                      """)

                else:
                    kmtrage = float(user_input)
                    if kmtrage < 0:
                        print("negative input detected")
                    estimation = shared.theta0 * kmtrage + shared.theta1
                    if estimation < 0:
                        print("negative result detected")
                    print(f"estimated price = {estimation}")

        except EOFError:
            print()
            break
        except ValueError:
            print("invalid input, please enter a number")


if __name__ == "__main__":
    main()
