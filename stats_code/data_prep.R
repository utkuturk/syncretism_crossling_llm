library(tidyverse)

# Load the data
eng <- read_csv("./paper/data/eng_surp_att.csv")
tr_surp <- read_csv("./paper/data/tr_surp.csv")
tr_att <- read_csv("./paper/data/tr_att.csv")
rus_surp <- read_csv("./paper/data/rus_surp.csv")
rus_att <- read_csv("./paper/data/rus_att.csv")


eng$surprisal_gpt2 <- eng$surprisal_gpt2 / log(2)
eng$syn <- ifelse(eng$is_syn == "yes", "no", "yes")
eng <- eng %>%
    select(-is_syn) %>%
    mutate(is_syn = syn) %>%
    select(-syn)

# View(eng)
eng_gpt_head_att <- eng %>% group_by(NP_Number, Auxiliary, is_syn) %>%
    summarize(
        mean = mean(attention_to_2nd_gpt),
        se = sd(attention_to_2nd_gpt) / sqrt(n())
    ) %>% mutate(model = "gpt", type ="head")

eng_gpt_attr_att <- eng %>%
    group_by(NP_Number, Auxiliary, is_syn) %>%
    summarize(
        mean = mean(attention_to_5th_gpt),
        se = sd(attention_to_5th_gpt) / sqrt(n())
    ) %>% mutate(model = "gpt", type ="attractor")


eng_bert_head_att <- eng %>%
    group_by(NP_Number, Auxiliary, is_syn) %>%
    summarize(
        mean = mean(attention_to_2nd_bert),
        se = sd(attention_to_2nd_bert) / sqrt(n())
    ) %>% mutate(model = "bert", type ="head")

eng_bert_attr_att <- eng %>%
    group_by(NP_Number, Auxiliary, is_syn) %>%
    summarize(
        mean = mean(attention_to_5th_bert),
        se = sd(attention_to_5th_bert) / sqrt(n())
    ) %>% mutate(model = "bert", type ="attractor")



eng_att_avg <- bind_rows(eng_gpt_head_att, eng_gpt_attr_att, eng_bert_head_att, eng_bert_attr_att) %>%
    mutate(
        attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
        verb_num = ifelse(Auxiliary == "is", "sg", "pl"),
        head_num = "sg",
        lang = "eng"
    ) %>%
    ungroup() %>%
    select(lang, head_num, attr_num, verb_num, model, is_syn, type, mean, se)

eng_gpt_surp <- eng %>%
    group_by(NP_Number, Auxiliary, is_syn) %>%
    summarize(
        mean = mean(surprisal_gpt2),
        se = sd(surprisal_gpt2) / sqrt(n())
    ) %>%
    mutate(model = "gpt")


eng_bert_surp <- eng %>%
    group_by(NP_Number, Auxiliary, is_syn) %>%
    summarize(
        mean = mean(surprisal_bert),
        se = sd(surprisal_bert) / sqrt(n())
    ) %>%
    mutate(model = "bert")

eng_surp_avg <- bind_rows(eng_gpt_surp, eng_bert_surp) %>%
    mutate(attr_num = ifelse(NP_Number == "singular", "sg", "pl"),
           verb_num = ifelse(Auxiliary == "is", "sg", "pl"),
           head_num = "sg",
        lang = "eng"
    ) %>%
    ungroup() %>%
    select(lang, head_num, attr_num, verb_num, model, is_syn, mean, se)



# Turkish Data
# View(tr_att)

tr_att <- tr_att %>%
    mutate(is_syn = ifelse(grepl("^[A-Z]", sentence), "yes", "no"))

tr_att_attr <- tr_att %>%
    group_by(attr_num, verb_num, model, is_syn) %>%
    summarize(
        mean = mean(attention_to_attractor),
        se = sd(attention_to_attractor) / sqrt(n())
    ) %>% mutate(type = "attractor")

tr_att_head <- tr_att %>%
    group_by(attr_num, verb_num, model, is_syn) %>%
    summarize(
        mean = mean(attention_to_head),
        se = sd(attention_to_head) / sqrt(n())
    ) %>% mutate(type = "head")

tr_att_avg <- bind_rows(tr_att_attr, tr_att_head) %>%
    mutate(lang = "tr", head_num = "sg") %>%
    ungroup() %>% select(lang, head_num, attr_num, verb_num, model, is_syn, type, mean, se)



tr_surp <- tr_surp %>%
    mutate(is_syn = ifelse(grepl("^[A-Z]", sentence), "yes", "no"))

tr_surp_avg <- tr_surp %>% group_by(attr_num, verb_num, model, is_syn) %>%
    summarize(
        mean = mean(surprisal),
        se = sd(surprisal) / sqrt(n())
    ) %>%
    mutate(lang = "tr", head_num = "sg") %>%
    ungroup() %>% select(lang, head_num, attr_num, verb_num, model, is_syn, mean, se)



# Russian Data

# View(rus_att)


rus_att <- rus_att %>%
    mutate(is_syn = ifelse(item <= 40, "no", "yes"))


rus_att_attr <- rus_att %>%
    group_by(head_num, attr_num, verb_num, is_syn) %>%
    summarize(
        mean = mean(attention_to_attractor),
        se = sd(attention_to_attractor) / sqrt(n())
    ) %>% mutate(type = "attractor")

rus_att_head <- rus_att %>%
    group_by(head_num, attr_num, verb_num, is_syn) %>%
    summarize(
        mean = mean(attention_to_head),
        se = sd(attention_to_head) / sqrt(n())
    ) %>% mutate(type = "head")

rus_att_avg <- bind_rows(rus_att_attr, rus_att_head) %>%
    mutate(head_num = ifelse(head_num == "SG", "sg", "pl"),
            attr_num = ifelse(attr_num == "SG", "sg", "pl"),
            verb_num = ifelse(verb_num == "SG", "sg", "pl"),
            lang = "rus", model = "bert") %>%
    ungroup() %>%
    select(lang, head_num, attr_num, verb_num, model, is_syn, type, mean, se)

rus_surp <- rus_surp %>% mutate(is_syn = ifelse(item <= 40, "no", "yes"))
rus_surp$surprisal <- rus_surp$surprisal / log(2)
# View(rus_surp)

rus_surp_avg <- rus_surp %>%
    group_by(head_num, attr_num, verb_num, is_syn) %>%
    summarize(
        mean = mean(surprisal),
        se = sd(surprisal) / sqrt(n())
    ) %>%
    mutate(
        head_num = ifelse(head_num == "SG", "sg", "pl"),
        attr_num = ifelse(attr_num == "SG", "sg", "pl"),
        verb_num = ifelse(verb_num == "SG", "sg", "pl"),
        lang = "rus", model = "bert"
    ) %>%
        ungroup() %>%
        select(lang, head_num, attr_num, verb_num, model, is_syn, mean, se)



att <- bind_rows(eng_att_avg, tr_att_avg, rus_att_avg)
surp <- bind_rows(eng_surp_avg, tr_surp_avg, rus_surp_avg)

# Save the results
write_csv(att, "./paper/data/att_avg.csv")
write_csv(surp, "./paper/data/surp_avg.csv")
