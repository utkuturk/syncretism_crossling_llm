library(tidyverse)
library(ggh4x)
library(patchwork)
library(ggforce)

att <- read.csv("./paper/data/att_avg.csv")
surp <- read.csv("./paper/data/surp_avg.csv")

surp$mean <- abs(surp$mean)
surp$model <- ifelse(surp$model == "bert", "BERT", "GPT-2")
surp$lang = ifelse(surp$lang == "eng", "English", ifelse(surp$lang == "tr", "Turkish", "Russian"))


## Surprisal By Language
surp <- surp %>%
mutate(is_syn = ifelse(is_syn =="yes", "Syncretic", "Non-syncretic"),
       match = ifelse(head_num == attr_num, "Match", "Mismatch"),
       attr_num = ifelse(attr_num == "pl", "Plural", "Singular")) %>%
       mutate(verb_num = ifelse(verb_num == "pl", "Ungrammatical", "Grammatical"), head_num = ifelse(head_num == "pl", "Plural Head", "Singular Head"))



diffs_with_se <- surp %>%
    group_by(lang, verb_num, model, is_syn, match) %>%
    summarise(mean = mean(mean), se = mean(se), .groups = "drop") %>%
    pivot_wider(
        names_from = match,
        values_from = c(mean, se),
        names_sep = "_"
    ) %>%
    mutate(
        diff = mean_Match - mean_Mismatch,
        se_diff = sqrt(se_Match^2 + se_Mismatch^2)
    )


surp_diff <- ggplot(diffs_with_se, aes(x = lang, y = diff, linetype = is_syn, shape = model)) +
    geom_point(position = position_dodge(width = 0.3), size = 2) +
    geom_errorbar(aes(ymin = diff - se_diff, ymax = diff + se_diff),
        width = 0.2, position = position_dodge(width = 0.3)
    ) +
    # add a line at 0 y
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    facet_nested(~verb_num, scales = "free_y") +
    theme_classic() +
    labs(
        x = "Language",
        y = expression(Delta * "Surprisal (Match - Mismatch)"),
        linetype = "Syncretism",
        shape = "Model"
    )

p_surp <- surp %>% filter(head_num == "Singular Head", verb_num == "Plural Verb") %>%
    ggplot(aes(x = match, y = mean, linetype = is_syn, group = is_syn)) +
    geom_point(aes(shape=is_syn), show.legend = F, position = position_dodge(width = 0.1), size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0, position = position_dodge(width = 0.1)) +
    facet_nested(model ~ verb_num + lang , scales = "free") +
    labs(x = "Head-Attractor Number", y = "Mean Surprisal", linetype = "Syncretism") +
    theme_classic()


p_surp_all <- surp %>%
    filter(head_num == "Singular Head") %>%
    ggplot(aes(x = match, y = mean, linetype = is_syn, group = is_syn)) +
    geom_point(aes(shape = is_syn), show.legend = F, position = position_dodge(width = 0.1), size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0, position = position_dodge(width = 0.1)) +
    facet_nested(model ~ verb_num + lang, scales = "free", independent="y") +
    labs(x = "Head-Attractor Number", y = "Mean Surprisal", linetype = "Syncretism") +
    theme_classic()


surp <- surp %>%
    mutate(lang_match = interaction(lang, match)) %>%
    mutate(lang_match = factor(lang_match, levels = c(
        "English.Match", "English.Mismatch",
        "Turkish.Match", "Turkish.Mismatch",
        "Russian.Match", "Russian.Mismatch"
    )))

p_surp_all_bar <-  surp %>%
    filter(head_num == "Singular Head") %>%
    ggplot(aes(x = is_syn, y = mean, fill = lang_match)) +
    geom_col(position = position_dodge(width = 0.8), color = "black") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se),
        position = position_dodge(width = 0.8), width = 0.2
    ) +
    facet_nested(model ~ verb_num + lang, scales = "free", independent="y") +
    labs(
        x = "Distinctive Case Marking",
        y = "Mean Surprisal",
        fill = "Language × Match"
    ) +
    theme_classic() +
    scale_fill_manual(
        values = c(
            "English.Match" = "#1f77b4",
            "English.Mismatch" = "#aec7e8",
            "Turkish.Match" = "#2ca02c",
            "Turkish.Mismatch" = "#98df8a",
            "Russian.Match" = "#d62728",
            "Russian.Mismatch" = "#ff9896"
        ),
        labels = c(
            "English.Match" = "English (Match)",
            "English.Mismatch" = "English (Mismatch)",
            "Turkish.Match" = "Turkish (Match)",
            "Turkish.Mismatch" = "Turkish (Mismatch)",
            "Russian.Match" = "Russian (Match)",
            "Russian.Mismatch" = "Russian (Mismatch)"
        )
    )

att$model <- ifelse(att$model == "bert", "BERT", "GPT-2")
att$lang <- ifelse(att$lang == "eng", "English", ifelse(att$lang == "tr", "Turkish", "Russian"))

att <- att %>%
    mutate(
        is_syn = ifelse(is_syn == "yes", "Syncretic", "Non-syncretic"),
        match = ifelse(head_num == attr_num, "Match", "Mismatch"),
        attr_num = ifelse(attr_num == "pl", "Plural", "Singular")
    ) %>%
    mutate(verb_num = ifelse(verb_num == "pl", "Ungrammatical", "Grammatical"), head_num = ifelse(head_num == "pl", "Plural Head", "Singular Head"))


diffs_with_se_head_att <- att %>%
    filter(head_num == "Singular Head", type == "head") %>%
    group_by(lang, verb_num, model, is_syn, match) %>%
    summarise(mean = mean(mean), se = mean(se), .groups = "drop") %>%
    pivot_wider(
        names_from = match,
        values_from = c(mean, se),
        names_sep = "_"
    ) %>%
    mutate(
        diff = mean_Match - mean_Mismatch,
        se_diff = sqrt(se_Match^2 + se_Mismatch^2)
    )


hatt_diff <- ggplot(diffs_with_se_head_att, aes(x = lang, y = diff, linetype = is_syn, shape = model)) +
    geom_point(position = position_dodge(width = 0.3), size = 2) +
    geom_errorbar(aes(ymin = diff - se_diff, ymax = diff + se_diff),
        width = 0.2, position = position_dodge(width = 0.3)
    ) +
    # add a line at 0 y
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    facet_nested(~verb_num, scales = "free_y") +
    theme_classic() +
    labs(
        x = "Language",
        y = expression(Delta * "Head Attention (Match - Mismatch)"),
        linetype = "Syncretism",
        shape = "Model"
    )

p_head_att <- att %>%
    filter(head_num == "Singular Head", verb_num == "Plural Verb", type == "head") %>%
    ggplot(aes(x = match, y = mean, linetype = is_syn, group = is_syn)) +
    geom_point(aes(shape = is_syn), show.legend = F, position = position_dodge(width = 0.1), size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0, position = position_dodge(width = 0.1)) +
    facet_nested(model ~ verb_num + lang, scales = "free", independent="y") +
    labs(x = "Head-Attractor Number", y = "Mean Attention to Head", linetype = "Syncretism") +
    theme_classic()


diffs_with_se_attractor_att <- att %>%
    filter(head_num == "Singular Head", type == "attractor") %>%
    group_by(lang, verb_num, model, is_syn, match) %>%
    summarise(mean = mean(mean), se = mean(se), .groups = "drop") %>%
    pivot_wider(
        names_from = match,
        values_from = c(mean, se),
        names_sep = "_"
    ) %>%
    mutate(
        diff = mean_Match - mean_Mismatch,
        se_diff = sqrt(se_Match^2 + se_Mismatch^2)
    )


aatt_diff <- ggplot(diffs_with_se_attractor_att, aes(x = lang, y = diff, linetype = is_syn, shape = model)) +
    geom_point(position = position_dodge(width = 0.3), size = 2) +
    geom_errorbar(aes(ymin = diff - se_diff, ymax = diff + se_diff),
        width = 0.2, position = position_dodge(width = 0.3)
    ) +
    # add a line at 0 y
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    facet_nested(~verb_num, scales = "free_y") +
    theme_classic() +
    labs(
        x = "Language",
        y = expression(Delta * "Attractor Attention (Match - Mismatch)"),
        linetype = "Syncretism",
        shape = "Model"
    )

p_attractor_att <- att %>%
    filter(head_num == "Singular Head", verb_num == "Plural Verb", type == "attractor") %>%
    ggplot(aes(x = match, y = mean, linetype = is_syn, group = is_syn)) +
    geom_point(aes(shape = is_syn), show.legend = F, position = position_dodge(width = 0.1), size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0, position = position_dodge(width = 0.1)) +
    facet_nested(model ~ verb_num + lang, scales = "free", independent = "y") +
    labs(x = "Head-Attractor Number", y = "Mean Attention to Attractor", linetype = "Syncretism") +
    theme_classic()

p_head_att_all <- att %>%
    filter(head_num == "Singular Head", type == "head") %>%
    ggplot(aes(x = match, y = mean, linetype = is_syn, group = is_syn)) +
    geom_point(aes(shape = is_syn), show.legend = F, position = position_dodge(width = 0.1), size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0, position = position_dodge(width = 0.1)) +
    facet_nested(model ~ verb_num + lang, scales = "free", independent = "y") +
    labs(x = "Head-Attractor Number", y = "Mean Attention to Head", linetype = "Syncretism") +
    theme_classic()


p_attractor_att_all <- att %>%
    filter(head_num == "Singular Head", type == "attractor") %>%
    ggplot(aes(x = match, y = mean, linetype = is_syn, group = is_syn)) +
    geom_point(aes(shape = is_syn), show.legend = F, position = position_dodge(width = 0.1), size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0, position = position_dodge(width = 0.1)) +
    facet_nested(model ~ verb_num + lang, scales = "free", independent = "y") +
    labs(x = "Head-Attractor Number", y = "Mean Attention to Attractor", linetype = "Syncretism") +
    theme_classic()


p_surp
patchwork <- p_head_att +
    theme(legend.position = "none") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_y_continuous(labels = scales::label_comma()) +
    p_attractor_att +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    plot_annotation(tag_levels = "A")


# make the x axis 45 degree


# Save plots with high fidelity and
# without clipping issues
ggsave("./paper/surp.pdf", p_surp, width = 6, height = 2, dpi = "retina")

ggsave("./paper/head_att.pdf", p_head_att, width = 6, height = 2, dpi = "retina")

ggsave("./paper/attractor_att.pdf", p_attractor_att, width = 6, height = 2, dpi = "retina")

ggsave("./paper/patchwork.pdf", patchwork, width = 12, height = 3, dpi = "retina")


ggsave("./paper/surp_all.pdf", p_surp_all, width = 12, height = 3, dpi = "retina")
ggsave("./paper/head_att_all.pdf", p_head_att_all, width = 12, height = 3, dpi = "retina")
ggsave("./paper/attractor_att_all.pdf", p_attractor_att_all, width = 12, height = 3, dpi = "retina")
ggsave("./paper/p_surp_all_bar.pdf", p_surp_all_bar, width = 12, height = 3, dpi = "retina")
ggsave("./paper/p_surp_diff.pdf", surp_diff, width = 6, height = 3, dpi = "retina")
ggsave("./paper/p_hatt_diff.pdf", hatt_diff, width = 6, height = 3, dpi = "retina")
ggsave("./paper/p_aatt_diff.pdf", aatt_diff, width = 6, height = 3, dpi = "retina")
