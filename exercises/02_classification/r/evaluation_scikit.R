library(ggplot2)
library(dplyr)
library(viridis)
library(tidyr)


df_acc = read.csv("../data/scikit_accuracies.csv")
df_acc$features = factor(df_acc$features)
df_acc$class = factor(df_acc$class)
df_acc$classifier_type = factor(df_acc$classifier_type)

levels(df_acc$features) = list("Full image" = "Full image", "Horizontal profile & Vertical profile" = "Horizontal profile,Vertical profile", "Horizontal profile & Vertical profile & Full image" = "Horizontal profile,Vertical profile,Full image")
levels(df_acc$classifier_type) = list("K-Neighbors (K=3)" = "KNeighbors[k=3]", "K-Neighbors (K=5)" = "KNeighbors[k=5]", "Linear SVC" = "LinearSVC", "MLP" = "MLP")


stats = df_acc %>% group_by(classifier_type, features) %>% summarise(mean_accuracy = mean(accuracy), lowest_accuracy = min(accuracy), highest_accuracy = max(accuracy))


ggplot(df_acc, aes(x = class, y = accuracy, fill = features)) +
  geom_col(position = 'dodge', color = '#444444') +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::percent) +
  facet_wrap(vars(classifier_type), nrow = 2) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_fill_viridis(discrete = TRUE) +
  labs(y = "Accuracy", x = "Class", fill = "Features") +
  ggsave("../resources/scikit_overview.png", width = 20, units = 'cm', dpi = 'print')

df_times_wide = read.csv("../data/scikit_times.csv")
df_times_wide$features = factor(df_times_wide$features)
df_times_wide$classifier_type = factor(df_times_wide$classifier_type)
df_times_wide$prediction_time = df_times_wide$prediction_time  / 48000

levels(df_times_wide$features) = list("Full image" = "Full image", "Horizontal profile & Vertical profile" = "Horizontal profile,Vertical profile", "Horizontal profile & Vertical profile & Full image" = "Horizontal profile,Vertical profile,Full image")
levels(df_times_wide$classifier_type) = list("K-Neighbors (K=3)" = "KNeighbors[k=3]", "K-Neighbors (K=5)" = "KNeighbors[k=5]", "Linear SVC" = "LinearSVC", "MLP" = "MLP")


df_times = gather(df_times_wide, stage, duration, training_time, prediction_time)


df_predictions = df_times %>% filter(stage == 'prediction_time')

ggplot(df_predictions, aes(x = classifier_type, y = duration, fill = features)) +
  geom_col(position = 'dodge', color = '#444444') +
  scale_x_discrete() +
  scale_y_log10() +
  # facet_wrap(vars(classifier_type), nrow = 2) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_fill_viridis(discrete = TRUE) +
  labs(y = "Prediction Duration [s / sample]", x = "Classifier", fill = "Features") +
  ggsave("../resources/scikit_prediction_timings.png", width = 20, units = 'cm', dpi = 'print')


df_training = df_times %>% filter(stage == 'training_time')

ggplot(df_training, aes(x = classifier_type, y = duration, fill = features)) +
  geom_col(position = 'dodge', color = '#444444') +
  scale_x_discrete() +
  scale_y_log10() +
  # facet_wrap(vars(classifier_type), nrow = 2) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_fill_viridis(discrete = TRUE) +
  labs(y = "Training Duration [s]", x = "Classifier", fill = "Features") +
  ggsave("../resources/scikit_training_timings.png", width = 20, units = 'cm', dpi = 'print')


