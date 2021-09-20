test = {
    "name": "ik_q2",
    "points": 1.5,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> import pickle
                    >>> import sys
                    >>> link1 = RevoluteDH(d=0.8, a=0, alpha=np.pi/2)
                    >>> link2 = RevoluteDH(d=0, a=0.8, alpha=0)
                    >>> link3 = RevoluteDH(d=0, a=0.8, alpha=0)
                    >>> my_bot = DHRobot([link1, link2, link3], name='3dof-manipulator')
                    >>> my_file = open("IK_q2.pk", "rb")
                    >>> data = pickle.load(my_file)
                    >>> target = SE3(data['target'])
                    >>> my_bot.q = data['initial_q']
                    >>> expected_q = data['q_sequence']
                    >>> sys.stdout.write('skip '); predicted_q = inverse_kinematics(my_bot, target, max_iterations=100, delta=0.1) # doctest:+ELLIPSIS
                    skip ...
                    >>> np.all(np.isclose(expected_q, predicted_q))
                    True
                    """,
                    "hidden": False,
                    "locked": False,
                }
            ],
            "scored": False,
            "setup": "",
            "teardown": "",
            "type": "doctest"
        }
    ]
}
