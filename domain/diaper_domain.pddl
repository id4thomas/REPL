(define (domain diaper_domain)
    (:types  character product element - object
    step literal - element
		)
    (:predicates
        (character ?character - character)
        (instock ?product - product)
        (purchased ?character - character ?product - product)
        (refunded ?character - character ?product - product)
        (have ?character - character ?product - product)
        (= ?o1 ?o2 - object)
        (need ?character - character ?product - product)
        )

    (:action purchase
        :parameters (?character - character ?product - product)
        :precondition (and (instock ?product)
                      (need ?character ?product)
                      (not (purchased ?character ?product)))

        :effect(and (have ?character ?product)
                    (not (instock ?product))
                    (purchased ?character ?product)))

    (:action find_spare
      :parameters (?character - character ?product - product)
      :precondition (and (have ?character ?product)
                          (need ?character ?product)
                          )
      :effect (not (need ?character ?product))
    )

    (:action refund
        :parameters (?character - character ?product - product)
        :precondition (and (have ?character ?product)
                           (not (need ?character ?product)))
        :effect (and (instock ?product)
                    (refunded ?character ?product)
                    (not (have ?character ?product))))
)
