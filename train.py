import torch
import os
import random
import time
import shutil
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from other.data.processing import get_train_val_dataloaders
from other.data.datasets import CocoDataset, SyntheticDataset, MixedDataset, TripletTransform, base_transform, \
    apply_custom_transform
from other.models.models_handler import MODELS, count_parameters, estimate_vram_usage
from other.utils import (find_model_in_dir_or_path, find_last_model_in_tree, print_as_table,
                         create_new_model_trains_dir, save_history_plot)
from other.losses import (BoundaryLossCalculator,
                          BlurBoundaryLoss)
from other.losses_utils import ImageProcessor

if __name__ == '__main__':
    from other.parsing.train_args_parser import *

    print(f"\n{'=' * 100}\n")

    train_coco_dataset = CocoDataset(
        cocodataset_path=train_coco_data_dir,
        transform=TripletTransform(
            transform=apply_custom_transform
        ))

    synthetic_dataset = SyntheticDataset(
        length=None,
        test_safe_to_desk=False
    )

    val_coco_dataset = CocoDataset(
        cocodataset_path=val_coco_data_dir,
        transform=TripletTransform(
            transform=base_transform
        ))

    mixed_dataset = MixedDataset(train_coco_dataset, synthetic_dataset, scale_factor=1.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train_results/model_name
    model_trains_tree_dir = os.path.join(train_res_dir, model_name)

    model_dir, model_path = None, None
    last_weights_path = None


    # в чем цель если weights задан мы берем именно его, если не задан
    # мы смотрим дан ли create_new_model если не задан, мы ищем модель
    # если задан weights_load_from, то мы его путь присваиваем last_weights_path
    # если create_new_model не True , то мы фактически знаем в какой директории weight его ссылку и он точно last_weights_path
    # если не задан weights_load_from и при этом create_new_model тоже, то мы так или иначе создадим новый
    if weights_load_from is not None:
        # train_results/model_name/2025-02-19_12-00-00/res_10/weights.pt
        # finding the first found weight.pt by given folder or path
        last_weights_path = find_model_in_dir_or_path(weights_load_from)
    # weights_load_from -> None/False
    else:
        # create_new_model -> None/False
        if (create_new_model is None) or (not create_new_model):
            model_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)
            last_weights_path = model_path
            if model_path is None:
                print(f"Couldn't find any model in {model_trains_tree_dir} so new model will be created")

    train_dataloader, val_dataloader, train_seed, validation_sed = [None] * 4
    if last_weights_path is not None:
        checkpoint = torch.load(last_weights_path, weights_only=True)
        train_seed = checkpoint["train_seed"]
        validation_seed = checkpoint["validation_seed"]
        train_dataloader, val_dataloader, _, _ = get_train_val_dataloaders(train_dataset=mixed_dataset,
                                                                           val_dataset=val_coco_dataset,
                                                                           batch_size=batch_size,
                                                                           val_batch_size=val_batch_size,
                                                                           num_workers=num_workers,
                                                                           val_num_workers=val_num_workers,
                                                                           train_seed=train_seed,
                                                                           val_seed=validation_seed)

        # Загружает сохранённые веса и параметры модели.
        # state_dict - это словарь, содержащий все параметры (веса и смещения) нейронной сети
        model.load_state_dict(checkpoint['model_state_dict'])
        # Загружает состояния оптимизатора.
        # Оптимизатор сохраняет информацию о шагах обновления параметров модели, таких как скорости и моменты градиентов
        # (например, для Adam или RMSPROP).
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.lr = lr
        curr_run_start_global_epoch = checkpoint['epoch'] + 1

        print(f"Successfully loaded {last_weights_path} with optimizer {checkpoint['optimizer']}")
        print(f"Continuing training from epoch {curr_run_start_global_epoch} on {device} device")
    else:
        # если нет last_weights_path weights.pt не найден
        curr_run_start_global_epoch = 1
        print(f"New model of {str(type(model))} type has been created, will be trained on {device} device")

    print(f"Estimated [parameters: {count_parameters(model)}, vram: {estimate_vram_usage(model):.4f} GB (if float32)]")

    try:
        # model_dir -> # train_results/model_name/2025-02-19_12-00-00/res_10/, TOT DIR ГДЕ НАХОДИТСЯ WEIGHT
        # last_weights_path ->  train_results/model_name/2025-02-19_12-00-00/res_10/weights.pt
        # Если у нас сущ файлы loss_history/accuracy_history мы их загружаем, если нет
        # мы используем из записанного csv колонку global_epoch как индексацуию
        loss_history_table = pd.read_csv(os.path.join(model_dir, 'loss_history.csv'), index_col="global_epoch")
        accuracy_history_table = pd.read_csv(os.path.join(model_dir, 'accuracy_history.csv'),
                                             index_col="global_epoch")
    except:
        # !!!!!
        # если в загруженном weight.pt точнее в его директории нет, игнорируем seed-ы
        # если loss и accuracy есть, но нету weights.pt ничего не загужается все ок
        train_dataloader, val_dataloader, train_seed, validation_seed = get_train_val_dataloaders(
            train_dataset=mixed_dataset, val_dataset=val_coco_dataset, batch_size=batch_size,
            val_batch_size=val_batch_size, num_workers=num_workers, val_num_workers=val_num_workers)

        # index_col = global_epoch | train_loss
        loss_history_table = pd.DataFrame(columns=['global_epoch', 'train_loss'])
        # index_col = global_epoch | train_accuracy
        accuracy_history_table = pd.DataFrame(columns=['global_epoch', 'train_accuracy'])
        loss_history_table.set_index('global_epoch', inplace=True)
        accuracy_history_table.set_index('global_epoch', inplace=True)

        print(f"Checkpoints(for this run): {save_frames}")

    for epoch in range(1, do_epoches + 1):
        # curr_run_start_global_epoch = checkpoint['epoch'] + 1
        global_epoch = curr_run_start_global_epoch + epoch - 1
        print(f"\n{'=' * 100}\n")

        # running_loss = torch.scalar_tensor(0, device=device)
        # running_correct_count = torch.scalar_tensor(0, device=device)
        # running_whole_count = torch.scalar_tensor(0, device=device)
        running_loss = 0.0
        running_correct_count = 0.0
        running_whole_count = 0.0

        model.train()

        batch_idx, batch_count = 0, len(train_dataloader)

        for batch_idx, (batch_inputs, batch_targets, mask) in enumerate(
                tqdm(train_dataloader, desc=f"Training epoch: {global_epoch} ({epoch}\\{do_epoches})" + ' | ')):

            batch_inputs = batch_inputs.to(device).detach()
            batch_targets = batch_targets.to(device).detach()
            mask = mask.to(device).detach()
            out = model(batch_inputs)


            boundary_loss = BoundaryLossCalculator(device=device)(out, batch_targets, mask).mean()

            bce_loss = F.binary_cross_entropy_with_logits(out, batch_targets.unsqueeze(1), reduction='mean')


            # out -> torch.Size([b, 1, h, w]), batch_targets -> torch.Size([b, h, w])
            blur_boundary_loss = BlurBoundaryLoss()(out, batch_targets)

            loss = alpha  * boundary_loss +  beta * blur_boundary_loss + gamma * bce_loss
            print(f'\n alpha * boundary_loss: {alpha * boundary_loss.item()}')
            print(f'beta * blur_boundary_loss: {beta * blur_boundary_loss.item()}')
            print(f'gamma * bce_loss: {gamma * bce_loss.item()} \n')


            loss = loss / accumulation_steps

            # batch_samples_count = torch.tensor(mask.size(0), device=mask.device)
            batch_samples_count = mask.size(0)

            # finally epoch loss

            running_loss += loss.detach().item() * batch_samples_count * accumulation_steps
            running_whole_count += batch_samples_count

            binary_out = (ImageProcessor.binarize_array(out.detach()))
            binary_target = (ImageProcessor.binarize_array(batch_targets.unsqueeze(1).detach()))
            pred_correct = (binary_out == binary_target).int()
            # pred_correct -> torch.Size([b, 1, h, w])
            # running_correct_count += torch.sum(pred_correct).detach() * batch_samples_count / pred_correct.numel()
            running_correct_count += torch.sum(pred_correct).item() * batch_samples_count / pred_correct.numel()



            # **************************************************************************************
            # Для НЕГО МЫ ЗАРАНЕ ДЕЛИМ LOSS НА КОЛИЧЕСТВО accumulation_steps ***********************
            # **************************************************************************************
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                out = out.detach()
                loss = loss.detach()

        # running_loss вычесляется для всей эпохи
        # running_loss = (loss_b1 * b + loss_b2 * b)/ running_whole_count = 2b
        with torch.no_grad():
            running_loss = (running_loss / running_whole_count).item()
            accuracy = (running_correct_count / running_whole_count).item()

            row_loss_values = {
                'global_epoch': global_epoch,
                'train_loss': running_loss
            }
            row_acc_values = {
                'global_epoch': global_epoch,
                'train_accuracy': accuracy
            }
            if print_level > 0:
                time.sleep(0.25)
                print(f"Training | loss: {running_loss:.4f} | accuracy: {accuracy:.4f}")
                time.sleep(0.25)

            val_loss, val_acc = None, None
            if val_every != 0 and epoch % val_every == 0:
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                correct_count = 0.0
                whole_count = 0.0
                # val_loss = torch.scalar_tensor(0, device=device)
                # val_acc = torch.scalar_tensor(0, device=device)
                # correct_count = torch.scalar_tensor(0, device=device)
                # whole_count = torch.scalar_tensor(0, device=device)
                print()
                time.sleep(0.25)
                for (batch_inputs, mask, batch_targets) in tqdm(val_dataloader,
                                                                desc=f"Calculating validation scores: "):
                    batch_inputs = batch_inputs.to(device)
                    mask = mask.to(device)
                    batch_targets = batch_targets.to(device)
                    real_samples_count = mask.size(0)

                    out = model(batch_inputs)
                    boundary_loss = BoundaryLossCalculator(device=device)(out, batch_targets, mask).mean()

                    bce_loss = F.binary_cross_entropy_with_logits(out, batch_targets.unsqueeze(1), reduction='mean')

                    blur_boundary_loss = BlurBoundaryLoss()(out, batch_targets)

                    loss = alpha * boundary_loss + beta * blur_boundary_loss + gamma * bce_loss

                    running_loss += loss.item() * real_samples_count
                    # running_loss += loss.detach() * real_samples_count
                    whole_count += real_samples_count

                    binary_out, _ = (ImageProcessor.binarize_array(out.detach()))
                    binary_target, _ = (ImageProcessor.binarize_array(batch_targets.unsqueeze(1).detach()))
                    pred_correct = (binary_out == binary_target).int()

                    # running_correct_count += torch.sum(
                    #     pred_correct).detach() * real_samples_count / pred_correct.numel()
                    running_correct_count += torch.sum(
                        pred_correct).item() * real_samples_count / pred_correct.numel()


                val_loss = (running_loss / whole_count).item()
                val_acc = (running_correct_count / whole_count).item()

                # row_loss_values = {
                #     'global_epoch': global_epoch,
                #     'train_loss': running_loss
                # }
                # row_acc_values = {
                #     'global_epoch': global_epoch,
                #     'train_accuracy': accuracy
                # }
                # Метод .loc[] позволяет изменять или добавлять строки в pandas.DataFrame, используя индекс. В данном случае:
                # global_epoch — это индекс, по которому вы вставляете данные.
                # row_loss_values и row_acc_values — это словари, которые содержат новые данные для вставки.
                row_loss_values['val_loss'] = val_loss if val_loss is not None else np.nan
                row_acc_values['val_accuracy'] = val_acc if val_acc is not None else np.nan

                # мы на loss_history_table, accuracy_history_table добавляем так-же val_loss, train_loss график
                loss_history_table.loc[global_epoch] = row_loss_values
                accuracy_history_table.loc[global_epoch] = row_acc_values

            # *********************************************************************************************************
            # тут мы для каждой epoch : epoch % val_every == 0:, выводим информацию историю loss-ов
            # тут epoch % val_every соответсвует сохраненным validation_loss и информации, для
            # [0=loss, epoch%val_every - 2=loss, epoch % val_every -1=none, epoch % val_every=none]
            # *********************************************************************************************************
            if val_every != 0 and print_level > 1 and epoch % val_every == 0:
                # data = {'col1': [1, 2, 3, 4, None, 6], 'col2': [10, 20, 30, 40, None, 60]}
                # df = pd.DataFrame(data)
                # def print_as_table(dataframe):
                #     if len(dataframe) > 4:
                #         print(tabulate(dataframe.iloc[[0, -3, -2, -1], :].T.fillna("---"), headers='keys'))
                #     else:
                #         print(tabulate(dataframe.T.fillna("---"), headers='keys'))
                # print_as_table(df)
                #         0    3  4      5
                # ----  ---  ---  ---  ---
                # col1    1    4  ---    6
                # col2   10   40  ---   60
                print(f"\nLoss history")
                print_as_table(loss_history_table)
                print(f"\nAccuracy history")
                print_as_table(accuracy_history_table)

            if epoch in save_frames:
                if model_dir is None:
                    # model_dir -> # train_results/model_name/2025-02-19_12-00-00/res_10/, TOT DIR ГДЕ НАХОДИТСЯ WEIGHT
                    # для каждого epoch пересоздаем папку, точнее файлы в нем перезаписываем
                    model_dir, model_path = create_new_model_trains_dir(model_trains_tree_dir)
                    print(f"\nCreated {model_dir}")

                # model_dir -> # train_results/model_name/2025-02-19_12-00-00/res_10/, TOT DIR ГДЕ НАХОДИТСЯ WEIGHT
                # если сущ-вовал тогда будет last_weights_path
                # train_results/model_name/2025-02-19_12-00-00/res_10/weights.pt
                if last_weights_path is not None:
                    old = os.path.join(model_dir, "old")
                    os.makedirs(old, exist_ok=True)
                    shutil.copy(last_weights_path, old)
            #
            # model_dir -> # train_results/model_name/2025-02-19_12-00-00/res_10/
            loss_history_table.to_csv(os.path.join(model_dir, 'loss_history.csv'))
            accuracy_history_table.to_csv(os.path.join(model_dir, 'accuracy_history.csv'))

            if plot:
                save_history_plot(loss_history_table, 'global_epoch', 'Loss history', 'Epoch', 'Loss',
                                  os.path.join(model_dir, 'loss.png'))

                save_history_plot(accuracy_history_table, 'global_epoch', 'Accuracy history', 'Epoch', 'Accuracy',
                                  os.path.join(model_dir, 'accuracy.png'))

            torch.save({
                'train_seed': train_seed,
                'validation_seed': validation_seed,
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': type(optimizer).__name__,
                'optimizer_state_dict': optimizer.state_dict()

            }, model_path)
            print(f"\nModel saved (global epoch: {global_epoch}, checkpoint: {epoch})")

            #             для перезаписывания в old weight.pt
            # res_i создается только при отсутствии weight.pt
            last_weights_path = model_path

            # для последующей загрузки, так-как у нас уже есть weight.pt после первой эпохи
            #     ydict['model']['weights'] = None
            #     ydict['model']['create_new_model'] = False
            model_has_been_saved()
