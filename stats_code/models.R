library(tidyverse)
library(brms)

# DATA PREP
eng <- read_csv("./paper/data/eng_surp_att.csv")
tr_surp <- read_csv("./paper/data/tr_surp.csv")
tr_att <- read_csv("./paper/data/tr_att.csv")
rus_surp <- read_csv("./paper/data/rus_surp.csv")
rus_att <- read_csv("./paper/data/rus_att.csv")


eng$surprisal_gpt2 <- eng$surprisal_gpt2 / log(2)
eng$syn <- ifelse(eng$is_syn == "yes", "no", "yes")
eng <- eng %>% select(-is_syn) %>%
    mutate(is_syn = syn) %>%
    select(-syn)


tr_att <- tr_att %>%
    mutate(is_syn = ifelse(grepl("^[A-Z]", sentence), "yes", "no"))

tr_surp <- tr_surp %>%
    mutate(is_syn = ifelse(grepl("^[A-Z]", sentence), "yes", "no"))

rus_att <- rus_att %>%
    mutate(is_syn = ifelse(item <= 40, "no", "yes"))

rus_surp <- rus_surp %>% mutate(is_syn = ifelse(item <= 40, "no", "yes"))
rus_surp$surprisal <- rus_surp$surprisal / log(2)


######################################################
######################################################
######################################################

############# HEAD ATTENTION #########################

head_attention <- bind_rows(
    rus_att %>% filter(head_num == "SG") %>%
        select(item, sentence, attr_num, verb_num, attention = attention_to_head, is_syn) %>%
        mutate(
            attr_num = ifelse(attr_num == "SG", "sg", "pl"),
            verb_num = ifelse(verb_num == "SG", "sg", "pl"),
            lang = "rus", model = "bert", type = "head"
        ),
    tr_att %>%
        select(item, sentence, attr_num, verb_num, attention = attention_to_head, model, is_syn) %>%
        mutate(lang = "tr", type = "head"),
    eng %>%
        mutate(
            attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
            verb_num = ifelse(Auxiliary == "is", "sg", "pl")
        ) %>%
        select(item = Item, sentence = Full_Sentence, attr_num, verb_num, attention = attention_to_2nd_gpt, is_syn) %>%
        mutate(lang = "eng", type = "head", model = "gpt"),
    eng %>%
        mutate(
            attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
            verb_num = ifelse(Auxiliary == "is", "sg", "pl")
        ) %>%
        select(item = Item, sentence = Full_Sentence, attr_num, verb_num, attention = attention_to_2nd_bert, is_syn) %>%
        mutate(lang = "eng", type = "head", model = "bert")
)
head_attention$attr_num <- as.factor(head_attention$attr_num)
head_attention$verb_num <- as.factor(head_attention$verb_num)
head_attention$is_syn <- as.factor(head_attention$is_syn)
head_attention$lang <- as.factor(head_attention$lang)
head_attention$model <- ifelse(head_attention$model == "gpt2", "gpt", head_attention$model)
head_attention$match = ifelse(head_attention$attr_num == head_attention$verb_num, "match", "mismatch")
head_attention$match = as.factor(head_attention$match)
head_attention$model <- as.factor(head_attention$model)
head_attention <- head_attention %>%
    mutate(
        lang = factor(lang, levels = c("eng", "rus", "tr")),
        model = factor(model),
        is_syn = factor(is_syn),
        attr_num = factor(attr_num)
    )

contrasts(head_attention$verb_num) = contr.sum(2)/2
contrasts(head_attention$attr_num) = contr.sum(2)/2
contrasts(head_attention$model) = contr.sum(2)/2
contrasts(head_attention$is_syn) = contr.sum(2)/2
contrasts(head_attention$match) = contr.sum(2)/2

my_brms <- function(my_formula, my_data) {
    m <- brm(
        formula = my_formula,
        data = my_data,
        family = gaussian(),
        chains = 4, cores = 4, iter = 2000,
    )
    return(m)
}


brms_p <- function(fit, coef, dir = "less") {
    h <- hypothesis(fit, paste0(coef, ifelse(dir == "less", " < 0", " > 0")))
    prob <- h$hypothesis$Post.Prob
    if (dir == "greater") prob <- 1 - prob
    sprintf("$P(\\beta %s 0) = %.3f$", ifelse(dir == "less", "<", ">"), prob)
}

find_int <- function(fit, term) {
    terms <- rownames(fixef(fit))
    interaction_terms <- terms[grepl(":", terms)]
    interaction_terms[grepl(term, interaction_terms)]
}

report_int <- function(fit, term) {
    interactions <- find_int(fit, term)

    for (t in interactions) {
        h <- hypothesis(fit, paste0(t, " < 0"))
        prob <- h$hypothesis$Post.Prob
        cat(t, ": ", sprintf("$P(\\beta < 0) = %.3f$", prob), "\n")
    }
}

full_brms_p <- function(fit) {
    for (t in rownames(fixef(fit))) {
        h <- hypothesis(fit, paste0(t, " < 0"))
        prob <- h$hypothesis$Post.Prob
        star <- if (prob >= 0.89 | prob <= 0.11) "*" else ""
        cat(sprintf("%-40s  %-45s  %s\n", t, sprintf("$P(\\beta < 0) = %.3f$", prob), star))
    }
}



model_fixed_eng <- my_brms(
    attention ~ match * model + match * is_syn + (1 | item),
    head_attention %>% filter(verb_num == "sg" & lang == "eng")
)

model_fixed_rus <- my_brms(
    attention ~ match * is_syn + (1 | item),
    head_attention %>% filter(verb_num == "sg" & lang == "rus")
)

model_fixed_tr <- my_brms(
    attention ~ match * model + match * is_syn + (1 | item),
    head_attention %>% filter(verb_num == "sg" & lang == "tr")
)

brms_p(model_fixed_eng, "match1", dir = "less")
brms_p(model_fixed_rus, "match1", dir = "less")
brms_p(model_fixed_tr, "match1", dir = "less")
report_int(model_fixed_tr, "match")
report_int(model_fixed_eng, "match")
report_int(model_fixed_rus, "match")


model_fixed_eng <- my_brms(
    attention ~ match * model * is_syn + (1 | item),
    head_attention %>% filter(verb_num == "pl" & lang == "eng")
)

model_fixed_rus <- my_brms(
    attention ~ match * is_syn + (1 | item),
    head_attention %>% filter(verb_num == "pl" & lang == "rus")
)

model_fixed_tr <- my_brms(
    attention ~ match * model * is_syn + (1 | item),
    head_attention %>% filter(verb_num == "pl" & lang == "tr")
)

brms_p(model_fixed_eng, "match1", dir = "less")
brms_p(model_fixed_rus, "match1", dir = "less")
brms_p(model_fixed_tr, "match1", dir = "less")

report_int(model_fixed_tr, "match")
report_int(model_fixed_eng, "match")
report_int(model_fixed_rus, "match")

######################################################
######################################################
######################################################
######################################################


######### ATtention to Attractor #####################
attractor_attention <- bind_rows(
    rus_att %>% filter(head_num == "SG") %>%
        select(item, sentence, attr_num, verb_num, attention = attention_to_attractor, is_syn) %>%
        mutate(
            attr_num = ifelse(attr_num == "SG", "sg", "pl"),
            verb_num = ifelse(verb_num == "SG", "sg", "pl"),
            lang = "rus", model = "bert", type = "attractor"
        ),
    tr_att %>%
        select(item, sentence, attr_num, verb_num, attention = attention_to_attractor, model, is_syn) %>%
        mutate(lang = "tr", type = "attractor"),
    eng %>%
        mutate(
            attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
            verb_num = ifelse(Auxiliary == "is", "sg", "pl")
        ) %>%
        select(item = Item, sentence = Full_Sentence, attr_num, verb_num, attention = attention_to_5th_gpt, is_syn) %>%
        mutate(lang = "eng", type = "attractor", model = "gpt"),
    eng %>%
        mutate(
            attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
            verb_num = ifelse(Auxiliary == "is", "sg", "pl")
        ) %>%
        select(item = Item, sentence = Full_Sentence, attr_num, verb_num, attention = attention_to_5th_bert, is_syn) %>%
        mutate(lang = "eng", type = "attractor", model = "bert")
)



attractor_attention$attr_num <- as.factor(attractor_attention$attr_num)
attractor_attention$verb_num <- as.factor(attractor_attention$verb_num)
attractor_attention$is_syn <- as.factor(attractor_attention$is_syn)
attractor_attention$lang <- as.factor(attractor_attention$lang)
attractor_attention$model <- ifelse(attractor_attention$model == "gpt2", "gpt", attractor_attention$model)
attractor_attention$model <- as.factor(attractor_attention$model)
attractor_attention$match <- ifelse(attractor_attention$attr_num == attractor_attention$verb_num, "match", "mismatch")
attractor_attention$match <- as.factor(attractor_attention$match)
attractor_attention <- attractor_attention %>%
    mutate(
        lang = factor(lang, levels = c("eng", "rus", "tr")),
        model = factor(model),
        is_syn = factor(is_syn),
        attr_num = factor(attr_num)
    )


contrasts(attractor_attention$verb_num) <- contr.sum(2) / 2
contrasts(attractor_attention$attr_num) <- contr.sum(2) / 2
contrasts(attractor_attention$model) <- contr.sum(2) / 2
contrasts(attractor_attention$is_syn) <- contr.sum(2) / 2
contrasts(attractor_attention$match) <- contr.sum(2) / 2

model_fixed_eng <- my_brms(
    attention ~ match * model * is_syn + (1 | item),
    attractor_attention %>% filter(verb_num == "sg" & lang == "eng")
)

model_fixed_rus <- my_brms(
    attention ~ match * is_syn + (1 | item),
    attractor_attention %>% filter(verb_num == "sg" & lang == "rus")
)

model_fixed_tr <- my_brms(
    attention ~ match * model * is_syn + (1 | item),
    attractor_attention %>% filter(verb_num == "sg" & lang == "tr")
)

brms_p(model_fixed_eng, "match1", dir = "less")
brms_p(model_fixed_rus, "match1", dir = "less")
brms_p(model_fixed_tr, "match1", dir = "less")
report_int(model_fixed_tr, "match")
report_int(model_fixed_eng, "match")
report_int(model_fixed_rus, "match")


model_fixed_tr <- my_brms(
    attention ~ match * is_syn + (1 | item),
    attractor_attention %>% filter(verb_num == "sg" & lang == "tr" & model == "bert")
)

brms_p(model_fixed_tr, "match1", dir = "less")
report_int(model_fixed_tr, "match")



model_fixed_eng <- my_brms(
    attention ~ match * is_syn + (1 | item),
    attractor_attention %>% filter(verb_num == "pl" & lang == "eng" & model == "bert")
)

model_fixed_rus <- my_brms(
    attention ~ match * is_syn + (1 | item),
    attractor_attention %>% filter(verb_num == "pl" & lang == "rus")
)

brms_p(model_fixed_eng, "match1", dir = "less")
brms_p(model_fixed_eng, "is_syn1", dir = "less")
brms_p(model_fixed_rus, "match1", dir = "less")
brms_p(model_fixed_rus, "is_syn1", dir = "less")
report_int(model_fixed_eng, "match")
report_int(model_fixed_rus, "match")

######################################################
######################################################
######################################################



################ SURPRISAL ###########################

surp <- bind_rows(
    rus_surp %>% filter(head_num == "SG") %>%
        select(item, sentence, attr_num, verb_num, surprisal, is_syn) %>%
        mutate(
            attr_num = ifelse(attr_num == "SG", "sg", "pl"),
            verb_num = ifelse(verb_num == "SG", "sg", "pl"),
            lang = "rus", model = "bert"),
    tr_surp %>%
        select(item, sentence, attr_num, verb_num, surprisal, model, is_syn) %>%
        mutate(lang = "tr"),
    eng %>%
        mutate(
            attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
            verb_num = ifelse(Auxiliary == "is", "sg", "pl")
        ) %>%
        select(item = Item, sentence = Full_Sentence, attr_num, verb_num, surprisal = surprisal_gpt2, is_syn) %>%
        mutate(lang = "eng", model = "gpt"),
    eng %>%
        mutate(
            attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
            verb_num = ifelse(Auxiliary == "is", "sg", "pl")
        ) %>%
        select(item = Item, sentence = Full_Sentence, attr_num, verb_num, surprisal = surprisal_bert, is_syn) %>%
        mutate(lang = "eng", model = "bert")
)
surp$attr_num <- as.factor(surp$attr_num)
surp$verb_num <- as.factor(surp$verb_num)
surp$is_syn <- as.factor(surp$is_syn)
surp$lang <- as.factor(surp$lang)
surp$model <- ifelse(surp$model == "gpt2", "gpt", surp$model)
surp$model <- as.factor(surp$model)
surp$match <- ifelse(surp$attr_num == surp$verb_num, "match", "mismatch")
surp$match <- as.factor(surp$match)
surp <- surp %>%
    mutate(
        lang = factor(lang, levels = c("eng", "rus", "tr")),
        model = factor(model),
        is_syn = factor(is_syn),
        attr_num = factor(attr_num)
    )


contrasts(surp$verb_num) <- contr.sum(2) / 2
contrasts(surp$attr_num) <- contr.sum(2) / 2
contrasts(surp$model) <- contr.sum(2) / 2
contrasts(surp$is_syn) <- contr.sum(2) / 2
contrasts(surp$match) <- contr.sum(2) / 2




bert_full <- my_brms(
    surprisal ~ match * verb_num * lang * is_syn + (1 | item),
    surp %>% filter(model == "bert")
    )


full_brms_p(bert_full)

gpt_full_sg <- my_brms(
    surprisal ~ match * lang * is_syn + (1 | item),
    surp %>% filter(model == "gpt", verb_num == "sg")
)

full_brms_p(gpt_full_sg)


eng_gpt_pl <- my_brms(
    surprisal ~ match * is_syn + (1 | item),
    surp %>% filter(model == "gpt", verb_num == "pl", lang == "eng")
)

full_brms_p(eng_gpt_pl)


tr_gpt_pl <- my_brms(
    surprisal ~ match * is_syn + (1 | item),
    surp %>% filter(model == "gpt", verb_num == "pl", lang == "tr")
)

full_brms_p(tr_gpt_pl)
