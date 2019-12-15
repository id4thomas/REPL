(define (problem diaper_story)
    (:domain diaper_domain)

    (:objects a b - character
              d - product)

    (:init
        (instock d)
        (need a d)
        (need b d)
        )

    (:goal (and
          (purchased a d)
          (purchased b d)
          (have b d)
          )
    ))
