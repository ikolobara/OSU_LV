def task1():
    work_hours = int(input("Radni sati: "))
    hourly_wage = float(input("eura/h: "))
    print(f"Ukupno: {total_euro(work_hours, hourly_wage)} eura") 

def total_euro(work_hours, hourly_wage):
    return work_hours * hourly_wage